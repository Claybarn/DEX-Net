"""Frozen shuffled coverage schedules with equal examples per training pool."""
import json,math
from pathlib import Path
import numpy as np
from .utils import dump


def make_schedule(root,lengths,epochs=3,batch_size=256,seed=260930,special=None,share=0.,examples_per_pool=None):
    """special: optional {pool: row positions}; the first round(share*batch_size) rows of every
    batch from that pool are drawn (with replacement) from them, the rest follow the coverage passes. examples_per_pool overrides the per-epoch count
    (default: the largest pool's size), e.g. for a short fine-tuning schedule."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():raise ValueError('Refusing to overwrite schedule')
    if not lengths or min(lengths.values())<1 or epochs<1:raise ValueError('Empty coverage population')
    names=sorted(lengths);slots=math.ceil((examples_per_pool or max(lengths.values()))/batch_size);per_epoch=slots*batch_size
    order=np.lib.format.open_memmap(root/'pool_order.npy',mode='w+',dtype='int16',shape=(epochs*slots*len(names),))
    offsets=np.lib.format.open_memmap(root/'pool_offset.npy',mode='w+',dtype='int64',shape=order.shape)
    cursor=0
    for epoch in range(epochs):
        rng=np.random.default_rng(seed+epoch*100003)
        for slot in range(slots):
            ids=rng.permutation(len(names));end=cursor+len(names)
            order[cursor:end]=ids;offsets[cursor:end]=epoch*per_epoch+slot*batch_size;cursor=end
    for no,name in enumerate(names):
        n=lengths[name];a=np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype='int32',shape=(epochs*per_epoch,))
        if n>=2**31:raise ValueError('Index dtype capacity exceeded')
        for epoch in range(epochs):
            rng=np.random.default_rng(seed+1000003*(no+1)+epoch);offset=epoch*per_epoch;end=offset+per_epoch
            while offset<end:
                permutation=rng.permutation(n).astype(np.int32);take=min(n,end-offset)
                a[offset:offset+take]=permutation[:take];offset+=take
        if special and len(special.get(name,[])):
            k=round(share*batch_size);pool=np.asarray(special[name],np.int32);rng=np.random.default_rng(seed+7919*(no+1))
            starts=np.arange(0,len(a),batch_size)
            a[(starts[:,None]+np.arange(k)[None]).ravel()]=pool[rng.integers(0,len(pool),len(starts)*k)]
        a.flush()
    order.flush();offsets.flush()
    result=dict(names=names,lengths=lengths,epochs=epochs,batch_size=batch_size,examples_per_pool_per_epoch=per_epoch,
                total_batches=len(order),total_examples=len(order)*batch_size,seed=seed,
                coverage='Each pool starts a new shuffled without-replacement pass each epoch; small pools repeat shuffled passes to equalize pool counts.'
                +(f' {share:.0%} of every batch is drawn with replacement from a special row set.' if special else ''),
                special_share=share if special else 0.,special_rows={n:int(len(v)) for n,v in (special or {}).items()})
    dump(root/'COMPLETE.json',result);return result


class CoverageSchedule:
    def __init__(self,root):
        root=Path(root);self.spec=json.loads((root/'COMPLETE.json').read_text());self.names=self.spec['names']
        self.order=np.load(root/'pool_order.npy',mmap_mode='r');self.offsets=np.load(root/'pool_offset.npy',mmap_mode='r')
        self.rows={n:np.load(root/(n+'.npy'),mmap_mode='r') for n in self.names}
    def __len__(self):return len(self.order)
    def __getitem__(self,i):
        name=self.names[int(self.order[i])];offset=int(self.offsets[i]);return name,self.rows[name][offset:offset+self.spec['batch_size']]
