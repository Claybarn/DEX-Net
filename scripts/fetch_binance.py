"""Download Binance spot 1-second klines and build a (timestamp, price) Parquet file.

Monthly archives come from data.binance.vision and are checked against Binance's
published SHA-256 before use. timestamp is the kline close time in Unix seconds
(open time + 1 s); price is the close. Binance switched spot timestamps from
milliseconds to microseconds in 2025; both are handled.

  python -m scripts.fetch_binance --symbol ETHUSDT --months 2025-09:2026-09 --out data/cex
"""
import argparse,hashlib,io,urllib.request,zipfile
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.csv as pacsv,pyarrow.parquet as pq

BASE='https://data.binance.vision/data/spot/monthly/klines'


def months(span):
    a,b=span.split(':');y,m=map(int,a.split('-'));y1,m1=map(int,b.split('-'))
    while (y,m)<=(y1,m1):
        yield f'{y:04d}-{m:02d}';y,m=(y+(m==12),m%12+1)


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'dexnet/0.1'}),timeout=300) as r:return r.read()


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--symbol',default='ETHUSDT');p.add_argument('--months',required=True,help='YYYY-MM:YYYY-MM inclusive')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.out/'raw';raw.mkdir(parents=True,exist_ok=True);ts,px=[],[]
    for month in months(a.months):
        name=f'{a.symbol}-1s-{month}.zip';path=raw/name
        if not path.exists():
            data=fetch(f'{BASE}/{a.symbol}/1s/{name}');expected=fetch(f'{BASE}/{a.symbol}/1s/{name}.CHECKSUM').decode().split()[0]
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
            path.write_bytes(data)
        with zipfile.ZipFile(path) as z:
            (inner,)=z.namelist();t=pacsv.read_csv(io.BytesIO(z.read(inner)),read_options=pacsv.ReadOptions(autogenerate_column_names=True))
        opened=t.column(0).to_numpy().astype(np.float64);unit=1e6 if opened[0]>1e14 else 1e3
        ts.append(opened/unit+1.);px.append(t.column(4).to_numpy().astype(np.float64));print(name,len(opened),flush=True)
    ts,px=np.concatenate(ts),np.concatenate(px);o=np.argsort(ts,kind='stable');ts,px=ts[o],px[o]
    if not (np.all(np.diff(ts)>0) and np.all(px>0)):raise ValueError('Unordered or invalid prices')
    out=a.out/f'{a.symbol.lower()}_1s.parquet';pq.write_table(pa.table(dict(timestamp=ts,price=px)),out,compression='zstd')
    print(out,len(ts),'rows; largest gap',float(np.diff(ts).max()),'s')


if __name__=='__main__':main()
