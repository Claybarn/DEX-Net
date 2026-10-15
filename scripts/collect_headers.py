"""Resumable exact Ethereum block times, from AWS headers with rotating RPC checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

from dexnet.rpc import RotatingRPC, json_post

BUCKET = 'aws-public-blockchain'
PREFIX = 'v1.0/eth/blocks/'


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def normalize(frame):
    """Keep exact integer seconds; conflicting duplicates are always fatal."""
    frame = frame.copy()
    if 'number' in frame:
        frame = frame.rename(columns={'number': 'block_id'})
    if not pd.api.types.is_integer_dtype(frame.timestamp):
        times = pd.to_datetime(frame.timestamp, utc=True)
        if np.any(times.astype('int64') % 1_000_000_000):
            raise ValueError('Non-integer timestamp')
        frame['timestamp'] = times.astype('int64') // 1_000_000_000
    for key in ['hash', 'parent_hash']:
        frame[key] = frame[key].map(lambda v: bytes.fromhex(v.removeprefix('0x')) if isinstance(v, str) else bytes(v))
        if not frame[key].map(len).eq(32).all():
            raise ValueError('Invalid block hash')
    frame = frame[['block_id', 'timestamp', 'hash', 'parent_hash']].drop_duplicates()
    if frame.block_id.duplicated().any():
        raise ValueError('Conflicting duplicate block')
    frame = frame.sort_values('block_id').reset_index(drop=True)
    if len(frame) > 1:
        if not (np.diff(frame.timestamp) > 0).all():
            raise ValueError('Non-increasing block time')
        consecutive = np.diff(frame.block_id) == 1
        links = frame.parent_hash.iloc[1:].to_numpy() == frame.hash.iloc[:-1].to_numpy()
        if not links[consecutive].all():
            raise ValueError('Broken parent hash chain')
    return frame


def utc_features(frame):
    frame = frame.copy()
    t = pd.to_datetime(frame.timestamp, unit='s', utc=True)
    seconds = (frame.timestamp % 86400).astype('int32')
    frame['utc_datetime'] = t
    frame['utc_second_of_day'] = seconds
    frame['utc_day_of_week'] = t.dt.dayofweek.astype('int8')
    frame['time_of_day_sin'] = np.sin(2 * np.pi * seconds / 86400).astype('float32')
    frame['time_of_day_cos'] = np.cos(2 * np.pi * seconds / 86400).astype('float32')
    return frame


def rpc_headers(rpc, blocks):
    payload = [dict(jsonrpc='2.0', id=int(b), method='eth_getBlockByNumber', params=[hex(int(b)), False]) for b in blocks]
    response = rpc(payload)
    if isinstance(response, dict):
        response = [response]
    rows = []
    for item in response:
        h = item['result']
        number = int(h['number'], 16)
        if number != item['id']:
            raise ValueError('RPC returned wrong block identity')
        rows.append(dict(block_id=number, timestamp=int(h['timestamp'], 16), hash=h['hash'], parent_hash=h['parentHash']))
    result = normalize(pd.DataFrame(rows))
    if set(result.block_id) != set(blocks):
        raise ValueError('RPC missing requested blocks')
    return result


def day_keys(day):
    keys, token = [], None
    while True:
        query = {'list-type': '2', 'prefix': PREFIX + 'date=' + day + '/'}
        if token:
            query['continuation-token'] = token
        url = 'https://' + BUCKET + '.s3.us-east-2.amazonaws.com/?' + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=45) as response:
            root = ET.fromstring(response.read())
        ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        keys.extend(n.text for n in root.findall('s:Contents/s:Key', ns) if n.text.endswith('.parquet'))
        token = root.findtext('s:NextContinuationToken', namespaces=ns)
        if not token:
            return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-block', type=int, default=12369621, help='default: Uniswap v3 factory deployment')
    p.add_argument('--last-block', type=int, default=25983637, help='default: end of the DEX-Net collection')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-rpc-gap-blocks', type=int, default=100000)
    p.add_argument('--max-seconds', type=int, default=21600)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / 'days'; cache.mkdir(exist_ok=True)
    started = time.monotonic()
    lo, hi = a.first_block, a.last_block
    if not 0 < lo <= hi: raise ValueError('Invalid block range')
    rpc = RotatingRPC(interval=.3)
    approved = []
    for url in rpc.urls:
        try:
            chain = json_post(url, dict(jsonrpc='2.0', id=1, method='eth_chainId', params=[]), timeout=30)
            if int(chain['result'], 16) != 1:
                raise ValueError('Provider is not Ethereum mainnet')
            approved.append(url)
        except (RuntimeError, KeyError) as exc:
            print('A provider was unavailable during the chain check:', type(exc).__name__, flush=True)
    if not approved:
        raise RuntimeError('No verified Ethereum mainnet provider')
    rpc.urls = approved; rpc.ready = [0.] * len(approved); rpc.failures = [0] * len(approved)
    endpoints = rpc_headers(rpc, [lo, hi])
    first_day, last_day = [datetime.fromtimestamp(int(t), timezone.utc).date() for t in endpoints.timestamp]
    dates = [(first_day + timedelta(days=i)).isoformat() for i in range((last_day-first_day).days+1)]
    manifest = dict(chain_id=1, first_block=lo, last_block=hi, total_blocks=hi-lo+1,
                    first_utc_day=str(first_day), last_utc_day=str(last_day), dates=len(dates),
                    source='s3://aws-public-blockchain/v1.0/eth/blocks/', rpc_sources=len(approved),
                    timestamp_method='exact header timestamp; no interpolation',
                    causal_feature_rule='Use the latest observed block timestamp only; future historical block times are labels, never rollout inputs.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    existing_manifest = a.out / 'manifest.json'
    if existing_manifest.exists() and json.loads(existing_manifest.read_text()) != manifest:
        raise ValueError('Existing run manifest differs; use a new output folder')
    write_json(existing_manifest, manifest)
    s3 = fs.S3FileSystem(anonymous=True, region='us-east-2', request_timeout=45, connect_timeout=15)

    def fetch(day):
        dest = cache / (day + '.parquet')
        if dest.exists():
            frame = normalize(pq.read_table(dest).to_pandas())
            return day, len(frame), None
        failure = None
        for attempt in range(4):
            try:
                keys = day_keys(day)
                if not keys:
                    return day, 0, 'no AWS files'
                tables = [pq.ParquetFile(BUCKET+'/'+key, filesystem=s3).read(columns=['number','timestamp','hash','parent_hash']) for key in keys]
                frame = normalize(pa.concat_tables(tables).to_pandas())
                frame = frame[(frame.block_id >= lo) & (frame.block_id <= hi)]
                if not (pd.to_datetime(frame.timestamp, unit='s', utc=True).dt.strftime('%Y-%m-%d') == day).all():
                    raise ValueError('Header outside its advertised UTC day')
                tmp = dest.with_suffix('.tmp')
                pq.write_table(pa.Table.from_pandas(utc_features(frame), preserve_index=False), tmp, compression='zstd')
                tmp.replace(dest)
                write_json(dest.with_suffix('.source.json'), dict(keys=keys, rows=len(frame)))
                return day, len(frame), None
            except ValueError:
                raise
            except Exception as exc:
                failure = repr(exc)
                time.sleep(min(2 ** attempt, 8))
        return day, 0, failure

    completed, count, failures = 0, 0, []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(fetch, day):day for day in dates}
        for future in as_completed(futures):
            day, n, error = future.result()
            completed += 1; count += n
            if error:
                failures.append(dict(day=day, error=error))
            elapsed = time.monotonic()-started
            status = dict(stage='download', days_processed=completed, days_total=len(dates),
                          blocks_cached=count, blocks_total=hi-lo+1, elapsed_seconds=elapsed,
                          estimated_seconds_remaining=elapsed / completed * (len(dates)-completed), failures=failures)
            write_json(a.out / 'progress.json', status)
            if completed % 50 == 0 or completed == len(dates):
                print(json.dumps(status), flush=True)
            if elapsed > a.max_seconds:
                for pending in futures: pending.cancel()
                raise RuntimeError('Runtime bound reached; resume with identical command')

    # Validate the complete ordered chain across daily partitions, without loading it all.
    previous = None; gaps = []; anchors = []; total = 0; minimum = None; maximum = None
    for i, day in enumerate(dates):
        path = cache / (day+'.parquet')
        if not path.exists():
            continue
        frame = normalize(pq.read_table(path).to_pandas())
        if frame.empty:
            continue
        if previous is not None:
            frame = normalize(pd.concat([pd.DataFrame([previous]), frame], ignore_index=True))
            frame_new = frame.iloc[1:]
        else:
            frame_new = frame
            minimum = int(frame.block_id.iloc[0])
        numbers = frame.block_id.to_numpy()
        for j in np.flatnonzero(np.diff(numbers) > 1):
            gaps.append([int(numbers[j]+1), int(numbers[j+1]-1)])
        previous = frame.iloc[-1].to_dict(); maximum = int(numbers[-1]); total += len(frame_new)
        if i % 32 == 0 or i == len(dates)-1:
            anchors.extend([frame_new.iloc[0].to_dict(), frame_new.iloc[-1].to_dict()])
    if minimum is None:
        gaps = [[lo, hi]]
    else:
        if minimum > lo: gaps.insert(0, [lo,minimum-1])
        if maximum < hi: gaps.append([maximum+1,hi])
    write_json(a.out / 'gaps.json', gaps)
    missing = sum(end-start+1 for start,end in gaps)
    if missing > a.max_rpc_gap_blocks:
        raise RuntimeError(f'{missing} missing blocks exceeds bounded RPC repair limit; partial cache preserved')
    repair = a.out / 'rpc-gaps'; repair.mkdir(exist_ok=True)
    for begin,end in gaps:
        for start in range(begin,end+1,20):
            stop = min(start+19,end)
            dest = repair / f'{start}-{stop}.parquet'
            if not dest.exists():
                frame = rpc_headers(rpc,list(range(start,stop+1)))
                tmp = dest.with_suffix('.tmp'); pq.write_table(pa.Table.from_pandas(utc_features(frame),preserve_index=False),tmp,compression='zstd'); tmp.replace(dest)
    # Fold repaired headers into the UTC-day partitions while retaining RPC originals.
    repairs_by_day = {}
    for path in repair.glob('*.parquet'):
        frame = normalize(pq.read_table(path).to_pandas())
        for day, group in frame.groupby(pd.to_datetime(frame.timestamp,unit='s',utc=True).dt.strftime('%Y-%m-%d')):
            repairs_by_day.setdefault(day, []).append(group)
    for day, groups in repairs_by_day.items():
        dest = cache / (day+'.parquet')
        if dest.exists(): groups.insert(0,normalize(pq.read_table(dest).to_pandas()))
        frame = normalize(pd.concat(groups,ignore_index=True))
        tmp = dest.with_suffix('.tmp'); pq.write_table(pa.Table.from_pandas(utc_features(frame),preserve_index=False),tmp,compression='zstd'); tmp.replace(dest)
    # Validate RPC anchors against bulk source, then all repaired links in sorted stream.
    anchors.extend(endpoints.to_dict('records'))
    for start in range(0,len(anchors),20):
        group = pd.DataFrame(anchors[start:start+20]).drop_duplicates('block_id')
        canonical = rpc_headers(rpc,group.block_id.tolist()).set_index('block_id')
        for row in group.itertuples():
            other = canonical.loc[row.block_id]
            if row.hash != other['hash'] or row.timestamp != other.timestamp:
                raise ValueError('Canonical RPC anchor disagrees with bulk headers')
    parts = list(cache.glob('*.parquet'))
    parts.sort(key=lambda path:int(pq.read_table(path,columns=['block_id']).column(0)[0].as_py()))
    previous = None; final_count = 0
    for path in parts:
        frame = normalize(pq.read_table(path).to_pandas())
        if len(frame)>1 and np.any(np.diff(frame.block_id)!=1):
            raise RuntimeError('Unrepaired internal daily gap')
        if previous is not None:
            if int(frame.block_id.iloc[0]) != previous['block_id']+1:
                raise ValueError('Final coverage is not consecutive')
            normalize(pd.DataFrame([previous,frame.iloc[0].to_dict()]))
        previous = frame.iloc[-1].to_dict(); final_count += len(frame)
    if final_count != hi-lo+1 or int(previous['block_id']) != hi:
        raise ValueError('Final coverage mismatch')
    report = dict(status='complete', blocks=final_count, first_block=lo,last_block=hi,
                  partitions=len(parts),
                  rpc_anchor_checks=len(anchors), repaired_blocks=missing, elapsed_seconds=time.monotonic()-started,
                  output_bytes=sum(p.stat().st_size for p in parts), full_parent_hash_chain_validated=True,
                  providers_verified_chain_id=1, no_interpolation=True)
    write_json(a.out/'report.json',report); write_json(a.out/'progress.json',report)
    (a.out/'COMPLETE').write_text(datetime.now(timezone.utc).isoformat()+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
