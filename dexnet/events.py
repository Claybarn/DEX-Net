"""Pool-independent v3 event decoding, exact liquidity accounting and dataset API."""
import math
from pathlib import Path
import json
import numpy as np
from .tick_mechanics import MIN_TICK, MAX_TICK, Q96, sqrt_at_tick

FEATURES = ['log_gap_seconds','log_price_return','swap_log_virtual_base_ratio',
            'lp_log_active_liquidity_ratio','range_offset_asinh','log_range_width_price',
            'log_gap_blocks','same_block','range_contains_price','log_liquidity_change',
            'active_liquidity_present','amount_reference_valid']
TARGETS = ['swap_log_virtual_base_ratio','mint_log_active_liquidity_ratio',
           'burn_fraction_logit','burn_full','range_lower_offset_steps','range_width_steps',
           'log_available_liquidity_ratio','gap_seconds']
SCALE_FEATURES = [0,1,2,3,4,5,6,9]


def signed(word, bits):
    value = word - (1 << 256) if word >= (1 << 255) else word
    if not -(1 << (bits-1)) <= value < (1 << (bits-1)):
        raise ValueError('Invalid signed ABI word')
    return value


def unsigned(word, bits):
    if not 0 <= word < (1 << bits):
        raise ValueError('Invalid unsigned ABI word')
    return word


def decode(log, topic_names):
    name = topic_names[log['topics'][0]]
    data = log['data'][2:]
    if len(data) % 64:
        raise ValueError('Malformed ABI data')
    words = [int(data[i:i+64],16) for i in range(0,len(data),64)]
    expected = {'Initialize':(1,2),'Swap':(3,5),'Mint':(4,4),'Burn':(4,3)}[name]
    if (len(log['topics']),len(words)) != expected:
        raise ValueError('Unexpected event ABI shape')
    r = dict(event=name, block=int(log['blockNumber'],16), log_index=int(log['logIndex'],16),
             tx_index=int(log['transactionIndex'],16), transaction=log['transactionHash'],
             block_hash=log['blockHash'], owner='', lower=0, upper=0, amount=0,
             amount0=0, amount1=0, sqrt=0, tick=0, liquidity=0)
    if name == 'Initialize':
        r.update(sqrt=unsigned(words[0],160),tick=signed(words[1],24))
    elif name == 'Swap':
        r.update(amount0=signed(words[0],256),amount1=signed(words[1],256),
                 sqrt=unsigned(words[2],160),liquidity=unsigned(words[3],128),tick=signed(words[4],24))
    else:
        if int(log['topics'][1],16) >= 1 << 160:
            raise ValueError('Invalid owner address')
        v = words[1:] if name == 'Mint' else words
        r.update(owner=log['topics'][1][-40:].lower(),lower=signed(int(log['topics'][2],16),24),
                 upper=signed(int(log['topics'][3],16),24),amount=unsigned(v[0],128),amount0=v[1],amount1=v[2])
    return r


class LiquidityLedger:
    """Fixed-grid integer Fenwick tree plus owner/range balances."""
    def __init__(self, spacing):
        self.spacing=spacing; self.first=MIN_TICK//spacing
        self.tree=[0]*((MAX_TICK//spacing-self.first)+3)
        self.positions={}; self.ranges={}; self.initialized=False
        self.sqrt=0; self.tick=0; self.L=0

    def change(self,tick,delta):
        j=tick//self.spacing-self.first+1
        while j<len(self.tree):
            self.tree[j]+=delta; j+=j&-j

    def active(self,tick):
        j=tick//self.spacing-self.first+1; value=0
        while j>0:
            value+=self.tree[j]; j-=j&-j
        return value

    def apply(self,r):
        event=r['event']
        if event == 'Initialize':
            if self.initialized:
                raise ValueError('Repeated initialization')
            self.initialized=True; self.sqrt=r['sqrt']; self.tick=r['tick']
        elif not self.initialized:
            raise ValueError('Event before initialization')
        elif event == 'Swap':
            if self.active(r['tick']) != r['liquidity']:
                raise ValueError(f"Active liquidity mismatch at {r['block']}:{r['log_index']}")
            self.sqrt=r['sqrt']; self.tick=r['tick']; self.L=r['liquidity']
        else:
            lo,hi,amount=r['lower'],r['upper'],r['amount']
            if not MIN_TICK<=lo<hi<=MAX_TICK or lo%self.spacing or hi%self.spacing:
                raise ValueError('Invalid liquidity range')
            key=(r['owner'],lo,hi); pair=(lo,hi)
            delta=amount if event=='Mint' else -amount
            if self.positions.get(key,0)+delta<0 or self.ranges.get(pair,0)+delta<0:
                raise ValueError('Burn exceeds reconstructed owner/range inventory')
            for mapping,k in [(self.positions,key),(self.ranges,pair)]:
                value=mapping.get(k,0)+delta
                if value: mapping[k]=value
                else: mapping.pop(k,None)
            if delta:
                self.change(lo,delta); self.change(hi,-delta)
                if lo<=self.tick<hi:self.L+=delta
            if self.L<0 or self.active(self.tick)!=self.L:
                raise ValueError('Liquidity update mismatch')
        if not MIN_TICK<=self.tick<MAX_TICK:
            raise ValueError('Invalid current tick')
        # At an exact leftward crossing the stored tick can be boundary - 1.
        if not sqrt_at_tick(self.tick)<=self.sqrt<=sqrt_at_tick(self.tick+1):
            raise ValueError('Square-root price inconsistent with current tick')


def orientation(meta):
    # Economic base is WETH for stable/WETH pairs, WBTC for WBTC/WETH,
    # and USDC for USDC/USDT. No dollar peg is imposed on any token.
    base = 'WBTC' if 'WBTC' in meta['pair'] else ('WETH' if 'WETH' in meta['pair'] else 'USDC')
    return base


def event_features(r, before, ledger, meta, previous, available):
    """Post-event history plus event target; ratio denominators are pre-event."""
    L,sqrt,tick=before; base0=meta['base_is_token0']; sign=1 if base0 else -1
    is_swap=r['event']=='Swap'; amount_base=r['amount0'] if base0 else r['amount1']
    if is_swap and amount_base==0:return None
    if not is_swap and not r['amount']:return None
    kind=(0 if amount_base>0 else 1) if is_swap else (2 if r['event']=='Mint' else 3)
    gap=r['timestamp']-previous[1]; blocks=r['block']-previous[0]
    if gap<0 or blocks<0:raise ValueError('Nonmonotone event time')
    f=np.zeros(len(FEATURES),np.float64); fm=np.ones(len(FEATURES),bool)
    y=np.zeros(len(TARGETS),np.float64); ym=np.zeros(len(TARGETS),bool)
    f[0]=math.log1p(gap); f[1]=sign*2*math.log(ledger.sqrt/sqrt)
    f[6]=math.log1p(blocks); f[7]=blocks==0; f[10]=ledger.L>0; f[11]=L>0
    fm[[2,3,4,5,8,9]]=False
    if L>0 and ledger.L>0:
        f[9]=math.log(ledger.L)-math.log(L); fm[9]=True
    if is_swap:
        if L>0:
            log_reserve=math.log(L)+(math.log(Q96)-math.log(sqrt) if base0 else math.log(sqrt)-math.log(Q96))
            f[2]=math.log(abs(amount_base))-log_reserve; fm[2]=True
            y[0]=f[2];ym[0]=True
    else:
        lower,upper=r['lower'],r['upper']
        lower_distance=lower-tick if base0 else tick-upper
        f[4]=math.asinh(lower_distance*math.log1p(.0001)/.001)
        f[5]=math.log((upper-lower)*math.log1p(.0001));f[8]=lower<=tick<upper
        fm[[4,5,8]]=True
        if L>0:
            f[3]=math.log(r['amount'])-math.log(L);fm[3]=True
        y[4]=lower_distance/meta['tick_spacing'];y[5]=(upper-lower)//meta['tick_spacing'];ym[[4,5]]=True
        if kind==2 and L>0:y[1]=f[3];ym[1]=True
        if kind==3:
            if not 0<r['amount']<=available:raise ValueError('Invalid aggregate burn fraction')
            full=r['amount']==available;y[3]=float(full);ym[3]=True
            if not full:y[2]=math.log(r['amount'])-math.log(available-r['amount']);ym[2]=True
            if L>0:y[6]=math.log(available)-math.log(L);ym[6]=True
    y[7]=gap;ym[7]=True
    return f,fm,y,ym,kind,blocks


def eligible_indices(blocks, lo, hi, horizon=1, seq_len=64):
    if horizon<1:raise ValueError('Horizon must be positive')
    start=max(seq_len,int(np.searchsorted(blocks,lo)))
    end=int(np.searchsorted(blocks,hi))-horizon+1
    return np.arange(start,max(start,end),dtype=np.int64)


class PooledDataset:
    """One pool/split view. Train samplers should choose pools before rows.

    New schema: incompatible with old hard-coded WETH/10-tick model heads.
    Output targets retain physical coordinates; optional target normalization
    must include its density Jacobian in a future pooled model.
    """
    def __init__(self,root,pool,split,horizon=1,seq_len=64):
        root=Path(root)
        if not (root/'COMPLETE.json').exists():raise ValueError('Pooled preparation is incomplete')
        if seq_len!=json.loads((root/'dataset.json').read_text())['seq_len']:
            raise ValueError('History length differs from audited split indices')
        self.folder=root/'pools'/pool;self.seq_len=seq_len;self.horizon=horizon
        meta=json.loads((self.folder/'metadata.json').read_text())
        if split=='train' and meta['held_out']:raise ValueError('Held-out pools cannot supply training samples')
        self.rows=np.load(self.folder/f'indices/{split}-h{horizon}.npy',mmap_mode='r')
        self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','targets','target_mask','kind','block','timestamp','log_index']}
        self.metadata=np.asarray(meta['metadata_features'],dtype=np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=int(self.rows[i]);a=self.arrays
        return dict(x=np.array(a['features'][row-self.seq_len:row]),history_mask=np.array(a['feature_mask'][row-self.seq_len:row]),
                    history_kind=np.array(a['kind'][row-self.seq_len:row]),kind=int(a['kind'][row]),
                    next_x=np.array(a['features'][row]),next_mask=np.array(a['feature_mask'][row]),
                    targets=np.array(a['targets'][row]),target_mask=np.array(a['target_mask'][row]),metadata=self.metadata.copy(),
                    current_timestamp=int(a['timestamp'][row-1]),current_block=int(a['block'][row-1]),
                    row=row,end_row=row+self.horizon-1,pool=self.folder.name)
