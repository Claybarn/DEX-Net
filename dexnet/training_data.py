"""Memory-mapped pooled batches with exact causal burn support."""
import json
import numpy as np
from pathlib import Path
import torch
from .events import PooledDataset
from .tick_mechanics import MIN_TICK,MAX_TICK


class PooledTrainingDataset(PooledDataset):
    def __init__(self,root,cache,pool,split,depth=None):
        super().__init__(root,pool,split)
        # Optional causal liquidity-depth profile (scripts.prepare_depth): state before each event.
        self.depth=np.load(Path(depth)/pool/'depth.npy',mmap_mode='r') if depth else None
        folder=cache/pool;self.context={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['anchor','buy_log_cap','burn_id','lower_offsets','width_offsets','lower_versions']}
        self.lower=np.memmap(folder/'lower_values.bin',dtype='int32',mode='r');self.width=np.memmap(folder/'width_values.bin',dtype='int32',mode='r')
        self.gaps=np.load(self.folder/'gap_blocks.npy',mmap_mode='r')
        self.meta=json.loads((self.folder/'metadata.json').read_text())
    def __getitem__(self,i):
        v=super().__getitem__(i);row=v['row'];c=self.context;target=v['targets'];spacing=self.meta['tick_spacing']
        lower=round(c['anchor'][row]+target[4]);width=round(target[5]);bid=int(c['burn_id'][row]);lo=np.empty(0,np.int32);wi=lo
        if bid>=0:
            version=int(c['lower_versions'][bid]);start,end=c['lower_offsets'][version:version+2];lo=np.asarray(self.lower[start:end])
            start,end=c['width_offsets'][bid:bid+2];wi=np.asarray(self.width[start:end])
            assert lower in lo and width in wi
        if self.depth is not None:v['depth']=np.array(self.depth[row])
        v.update(anchor=float(c['anchor'][row]),buy_log_cap=float(c['buy_log_cap'][row]),dt=int(self.gaps[row]),lower=lower,width=width,
                 grid_min=-((-MIN_TICK)//spacing),grid_max=MAX_TICK//spacing,unit=spacing*np.log1p(.0001),lo=lo,wi=wi)
        return v


def pooled_collate(items):
    fields=['x','history_mask','history_kind','kind','targets','target_mask','metadata','anchor','buy_log_cap','dt','lower','width','grid_min','grid_max','unit']
    b={k:torch.as_tensor(np.stack([v[k] for v in items])) for k in fields}
    b['x']=b['x'].float();b['metadata']=b['metadata'].float();b['history_kind']=b['history_kind'].long();b['kind']=b['kind'].long()
    if 'depth' in items[0]:b['depth']=torch.as_tensor(np.stack([v['depth'] for v in items])).float()
    burns=[v for v in items if v['kind']==3]
    for key,dest in [('lo','lower'),('wi','width')]:
        size=max([len(v[key]) for v in burns],default=0);a=np.ones((len(burns),size),np.float32);mask=np.zeros_like(a,bool)
        for i,v in enumerate(burns):a[i,:len(v[key])]=v[key];mask[i,:len(v[key])]=True
        b[dest+'_candidates']=torch.from_numpy(a);b[dest+'_mask']=torch.from_numpy(mask)
    return b
