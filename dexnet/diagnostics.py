"""Exact pooled rollout feedback and finite-ensemble proper scores."""
import math
from types import SimpleNamespace
import numpy as np
from .events import event_features
from .model import native_range

HORIZONS=(1,8,32)
# Fixed physical transforms, not fitted on evaluation outcomes.
ENDPOINT_SCALES=np.array([10.,.1,1.,.001])
ENDPOINT_NAMES=('return_bp','log1p_liquidity_change','log1p_blocks','log1p_base_volume_over_initial_reserve')


def proper_scores(samples,target):
    """Unbiased ensemble energy score and marginal CRPS (off-diagonal pairs)."""
    x=np.asarray(samples,dtype=np.float64);y=np.asarray(target,dtype=np.float64);n=len(x)
    if n<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid score ensemble')
    pair=np.abs(x[:,None,:]-x[None,:,:]);crps=np.abs(x-y).mean(0)-pair.sum((0,1))/(2*n*(n-1))
    z=x/ENDPOINT_SCALES;v=y/ENDPOINT_SCALES
    energy=np.linalg.norm(z-v,axis=-1).mean()-np.linalg.norm(z[:,None]-z[None,:],axis=-1).sum()/(2*n*(n-1))
    return float(energy),crps


def endpoint(state,initial,blocks,volume,meta):
    sign=1 if meta['base_is_token0'] else -1
    reserve=math.log(initial.liquidity)+(math.log(1<<96)-math.log(initial.sqrt) if meta['base_is_token0'] else math.log(initial.sqrt)-math.log(1<<96))
    return [sign*2*math.log(state.sqrt/initial.sqrt)*1e4,math.log1p(state.liquidity)-math.log1p(initial.liquidity),math.log1p(blocks),math.log1p(volume*math.exp(-reserve))]


def feedback(previous,state,event,meta,mean,std,available=0,seconds_per_block=12.):
    """Use the same feature builder/scalers as audited real training examples."""
    r=dict(event='Swap' if event['kind']<2 else ('Mint' if event['kind']==2 else 'Burn'),block=event['dt'],timestamp=round(seconds_per_block*event['dt']),amount=0,amount0=0,amount1=0,lower=0,upper=0)
    if event['kind']<2:r['amount0' if meta['base_is_token0'] else 'amount1']=event['base_amount']
    else:r.update(amount=event['amount'],**dict(zip(['lower','upper'],native_range(event['lower'],event['upper'],meta))))
    post=SimpleNamespace(L=state.liquidity,sqrt=state.sqrt,tick=state.tick)
    f,mask,*_=event_features(r,(previous.liquidity,previous.sqrt,previous.tick),post,meta,(0,0),available)
    scaled=np.where(mask,(f-mean)/std,0).astype(np.float32)
    if not np.isfinite(scaled).all():raise ValueError('Nonfinite feedback features')
    return scaled,mask
