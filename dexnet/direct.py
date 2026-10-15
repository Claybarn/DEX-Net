"""Direct (nonrecursive) joint price/volume forecasts at fixed clock horizons: the paper's baseline.

The encoder is the event model's history Transformer; a noise-conditioned decoder
emits joint samples at 5 min, 30 min, 2 h and 6 h, trained with an energy/CRPS objective.
"""
import math
import numpy as np
import torch
from torch import nn
from .model import PooledEventModel
from .scores import energy_surrogate

HORIZONS=(300,1800,7200,21600)
TARGET_NAMES=('price_log_bp','log1p_volume_over_initial_virtual_reserve','log1p_sum_swap_virtual_reserve_ratios')


def future_labels(event_times,log_sqrt,log_reserve,volume,turnover,invalid,anchor_times,horizons=HORIZONS):
    """Complete-block outcomes; prefixes contain a leading zero. No future inputs."""
    start=np.searchsorted(event_times,anchor_times,side='right')-1
    end=np.searchsorted(event_times,anchor_times[:,None]+np.array(horizons)[None],side='right')-1
    safe=np.maximum(start,0);e=np.maximum(end,0)
    returns=2*(log_sqrt[e]-log_sqrt[safe,None])*1e4
    v=volume[e+1]-volume[safe[:,None]+1];turn=turnover[e+1]-turnover[safe[:,None]+1]
    y=np.stack([returns,np.log1p(np.maximum(v,0)*np.exp(-log_reserve[safe,None])),np.log1p(np.maximum(turn,0))],-1)
    good=(start>=0)&np.isfinite(log_reserve[safe])
    good &= ((invalid[e+1]-invalid[safe[:,None]+1])==0).all(1)
    good &= np.isfinite(y).all((1,2))
    return y,good,start,end


class DirectForecast(nn.Module):
    def __init__(self,scalers,scales,increments,d_model=128,layers=4,noise_dim=16,gas_dim=0,gas_scaler=None):
        super().__init__();self.backbone=PooledEventModel(scalers,d_model=d_model,layers=layers)
        for name,p in self.backbone.named_parameters():
            p.requires_grad_(name=='query' or name.startswith(('project.','metadata_project.','encoder.','norm.','kind_emb.','position.')))
        self.age=nn.Linear(2,d_model);nn.init.zeros_(self.age.weight);nn.init.zeros_(self.age.bias)
        self.decoder=nn.Sequential(nn.Linear(d_model+noise_dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,len(HORIZONS)*3))
        self.noise_dim=noise_dim
        self.gas_dim=gas_dim
        if gas_dim:
            if gas_scaler is None or len(gas_scaler['mean'])!=gas_dim or len(gas_scaler['std'])!=gas_dim:
                raise ValueError('Gas-enabled model requires its training scaler')
            self.gas_project=nn.Linear(gas_dim,d_model,bias=False)
            nn.init.zeros_(self.gas_project.weight)
            self.register_buffer('gas_mean',torch.tensor(gas_scaler['mean'],dtype=torch.float32))
            self.register_buffer('gas_std',torch.tensor(gas_scaler['std'],dtype=torch.float32))
            if not torch.isfinite(self.gas_std).all() or not (self.gas_std>0).all():raise ValueError('Invalid gas scaler')
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
        self.register_buffer('increments',torch.tensor(increments,dtype=torch.float32))
    def encode(self,b):
        h=self.backbone.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'])+self.age(b['age']/10.)
        if self.gas_dim:
            if 'gas' not in b:raise ValueError('Missing causal gas features')
            h=h+self.gas_project((b['gas']-self.gas_mean)/self.gas_std)
        return h
    def sample(self,b,paths=16,noise=None):
        h=self.encode(b)
        if noise is None:noise=torch.randn((len(h),paths,self.noise_dim),device=h.device,dtype=h.dtype)
        raw=self.decoder(torch.cat([h[:,None].expand(-1,paths,-1),noise],-1)).reshape(len(h),paths,len(HORIZONS),3)
        price=raw[...,0]*self.scales[:,0]
        # ReLU includes exact zero-volume probability; positive increments enforce horizon ordering.
        volumes=(raw[...,1:].relu()*self.increments).cumsum(-2)
        return torch.cat([price[...,None],volumes],-1)
    def loss(self,b,paths=16):
        samples=self.sample(b,paths);x=(samples/self.scales).flatten(2);target=(b['y']/self.scales).flatten(1)
        energy,_=energy_surrogate(x,target,torch.zeros(x.shape[:2],device=x.device))
        pair=(x[:,:,None]-x[:,None,:]).abs().sum((1,2))/(2*paths*(paths-1))
        marginal=((x-target[:,None]).abs().mean(1)-pair).mean()
        # Both are proper; joint term retains cross-horizon/target dependence.
        return .5*energy/math.sqrt(x.shape[-1])+.5*marginal


class DirectDataset:
    def __init__(self,data,labels,pool,split,gas_features=None):
        import json
        from pathlib import Path
        self.folder=Path(labels)/pool;self.meta=json.loads((Path(data)/'pools'/pool/'metadata.json').read_text())
        if split=='train' and self.meta['held_out']:raise ValueError('Heldout training forbidden')
        self.rows=np.load(self.folder/(split+'.npy'),mmap_mode='r')
        self.gas=np.load(Path(gas_features)/(pool+'.npy'),mmap_mode='r') if gas_features is not None else None
        self.fields={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in ['history_end','age','y','baseline']}
        self.history={k:np.load(Path(data)/'pools'/pool/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','kind']}
    def __len__(self):return len(self.rows)
    def batch(self,indices,device):
        selected=self.rows[indices];ends=self.fields['history_end'][selected];history=ends[:,None]+np.arange(-64,0)[None]
        batch=dict(x=torch.as_tensor(self.history['features'][history],device=device),history_mask=torch.as_tensor(self.history['feature_mask'][history],device=device),history_kind=torch.as_tensor(self.history['kind'][history],device=device,dtype=torch.long),metadata=torch.tensor(self.meta['metadata_features'],device=device,dtype=torch.float32).expand(len(selected),-1),age=torch.as_tensor(np.array(self.fields['age'][selected]),device=device),y=torch.as_tensor(np.array(self.fields['y'][selected]),device=device),baseline=torch.as_tensor(np.array(self.fields['baseline'][selected]),device=device))
        if self.gas is not None:batch['gas']=torch.as_tensor(np.array(self.gas[selected]),device=device)
        return batch


def load_direct_checkpoint(ck,device='cuda'):
    config={k:v for k,v in ck['model_config'].items() if k in ['d_model','layers','noise_dim','gas_dim']}
    model=DirectForecast(ck['scalers'],ck['scales']['scales'],ck['scales']['increments'],gas_scaler=ck.get('gas_scaler'),**config).to(device)
    model.load_state_dict(ck['model'])
    return model
