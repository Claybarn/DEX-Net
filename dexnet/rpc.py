"""JSON-RPC access with provider rotation.

Endpoints come from the DEXNET_RPC_URLS environment variable (comma-separated),
so no provider or API key is stored in the code.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request


def rpc_urls():
    urls = [u.strip() for u in os.environ.get('DEXNET_RPC_URLS', '').split(',') if u.strip()]
    if not urls:
        raise RuntimeError('Set DEXNET_RPC_URLS to one or more Ethereum JSON-RPC endpoints (comma-separated)')
    return urls


def json_post(url, payload, timeout=60):
    """One POST; raises RuntimeError on transport or HTTP errors (TLS always verified)."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json', 'User-Agent': 'dexnet/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'RPC request failed: {type(exc).__name__}') from None


class RotatingRPC:
    def __init__(self, urls=None, interval=1., cooldown=15.):
        self.urls = list(urls) if urls else rpc_urls()
        self.interval, self.cooldown = interval, cooldown
        self.lock = threading.Lock()
        self.active = 0
        self.ready = [0.] * len(self.urls)
        self.failures = [0] * len(self.urls)
        self.last_request = 0.

    def __call__(self, payload, attempts=8):
        # Serialize callers so they share rate limits and the successful provider.
        with self.lock:
            for _ in range(attempts):
                now = time.monotonic()
                order = [(self.active+i) % len(self.urls) for i in range(len(self.urls))]
                available = [i for i in order if self.ready[i] <= now]
                i = available[0] if available else min(order, key=lambda j:self.ready[j])
                delay = max(self.ready[i]-now, self.last_request+self.interval-now, 0.)
                if delay:
                    time.sleep(delay)
                self.last_request = time.monotonic()
                try:
                    result = json_post(self.urls[i], payload, timeout=30)
                    responses = result if isinstance(result, list) else [result]
                    expected = payload if isinstance(payload, list) else [payload]
                    by_id = {r.get('id'):r for r in responses if isinstance(r, dict)}
                    if len(responses) != len(expected) or any(
                        by_id.get(p['id'], {}).get('result') is None or
                        by_id.get(p['id'], {}).get('error') is not None for p in expected
                    ):
                        raise RuntimeError(f'Incomplete/error RPC response: {str(result)[:300]}')
                    self.active = i
                    self.failures[i] = 0
                    return result
                except RuntimeError as exc:
                    error = exc
                    self.failures[i] += 1
                    self.ready[i] = time.monotonic()+min(60., self.cooldown*2**(self.failures[i]-1))
                    self.active = (i+1) % len(self.urls)
                    print(f"RPC provider {i} failed ({exc}); rotating", flush=True)
            raise RuntimeError(f'All RPC providers exhausted: {error}')


def jsonrpc(url, method, params, attempts=8):
    """Call with retries on rate limits and transient errors.

    Request errors (e.g. a block range the provider rejects) raise ValueError
    carrying the provider's error object, so callers can adapt; URLs are never
    included in messages.
    """
    body = json.dumps(dict(jsonrpc='2.0', id=1, method=method, params=params)).encode()
    err = None
    for k in range(attempts):
        try:
            request = urllib.request.Request(url, body, {'Content-Type': 'application/json', 'User-Agent': 'dexnet/0.1'})
            r = json.loads(urllib.request.urlopen(request, timeout=120).read())
            if 'result' in r:
                return r['result']
            err = r.get('error')
            if err and err.get('code') in (-32602, -32600, 35):
                raise ValueError(f'{method}: {err}')
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}'
            try: err = json.loads(e.read()).get('error', err)
            except Exception: pass
            if e.code not in (429, 500, 502, 503, 504):
                raise ValueError(f'{method}: {err}') from None
        except (urllib.error.URLError, TimeoutError) as e:
            err = type(e).__name__
        time.sleep(min(30, 2**k))
    raise RuntimeError(f'{method} failed after retries: {err}')
