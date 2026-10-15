import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from dexnet.events import LiquidityLedger,decode,signed,event_features,eligible_indices
from dexnet.tick_mechanics import Q96,sqrt_at_tick


def event(name,**kwargs):
    return dict(event=name,block=2,log_index=1,timestamp=120,owner='a',lower=-60,upper=60,amount=1000,
                amount0=0,amount1=0,sqrt=Q96,tick=0,liquidity=1000,**kwargs)


class PooledDataTests(unittest.TestCase):
    def test_signed_word_and_abi(self):
        self.assertEqual(signed((1<<256)-60,24),-60)
        with self.assertRaises(ValueError):signed(1<<24,24)
        words=[10,(1<<256)-20,Q96,1000,0]
        log=dict(topics=['swap','0x'+'0'*64,'0x'+'0'*64],data='0x'+''.join(f'{x:064x}' for x in words),
                 blockNumber='0x2',logIndex='0x1',transactionIndex='0x0',transactionHash='tx',blockHash='hash')
        r=decode(log,{'swap':'Swap'});self.assertEqual((r['amount0'],r['amount1']),(10,-20))

    def test_liquidity_matches_swaps_owner_burns_and_boundary(self):
        for spacing in [1,10,60]:
            ledger=LiquidityLedger(spacing);ledger.apply(event('Initialize'));ledger.apply(event('Mint'))
            self.assertEqual(ledger.L,1000)
            r=event('Swap');r.update(tick=60,sqrt=sqrt_at_tick(60),liquidity=0);ledger.apply(r)
            r.update(tick=59,liquidity=1000);ledger.apply(r)
            burn=event('Burn');burn['owner']='wrong'
            with self.assertRaises(ValueError):ledger.apply(burn)
            ledger.apply(event('Burn'));self.assertEqual(ledger.L,0)
            with self.assertRaises(ValueError):ledger.apply(event('Swap'))

    def test_token_orientation_and_decimals_cancel_in_depth_ratio(self):
        ledger=LiquidityLedger(10);ledger.apply(event('Initialize'));ledger.apply(event('Mint'))
        r=event('Swap');r.update(amount0=100,amount1=-100)
        for base0,kind in [(True,0),(False,1)]:
            meta=dict(base_is_token0=base0,tick_spacing=10)
            f,fm,y,ym,k,g=event_features(r,(1000,Q96,0),ledger,meta,(1,108),0)
            self.assertEqual(k,kind);self.assertAlmostEqual(y[0],np.log(.1));self.assertTrue(ym[0]);self.assertEqual(g,1)
        r=event('Mint');f,fm,y,ym,_,_=event_features(r,(0,Q96,0),ledger,dict(base_is_token0=True,tick_spacing=10),(1,108),0)
        self.assertFalse(fm[3]);self.assertFalse(ym[1]);self.assertTrue(np.isfinite(y).all())

    def test_purge_horizons_and_block_boundaries(self):
        blocks=np.repeat(np.arange(100),3)
        rows=eligible_indices(blocks,25,60,horizon=32)
        self.assertTrue(np.all(blocks[rows]>=25));self.assertTrue(np.all(blocks[rows+31]<60))
        self.assertGreaterEqual(rows[0],64)
        self.assertEqual(len(eligible_indices(blocks,25,26,horizon=32)),0)

    def test_end_to_end_scalers_exclude_heldout_and_future(self):
        from scripts.prepare_data import finalize
        from dexnet.events import PooledDataset,FEATURES,TARGETS
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'headers').mkdir()
            times=1500000000+np.arange(300)*12;np.save(root/'headers/timestamp.npy',times)
            protocol=dict(first_block=100,end_block=399,calendar_splits=dict(train_before_block=200,validation_before_block=300,held_out_pools=['held']))
            metas=[dict(name=n) for n in ['a','b','held']]
            for i,meta in enumerate(metas):
                folder=root/'pools'/meta['name'];folder.mkdir(parents=True)
                (folder/'metadata.json').write_text(json.dumps(dict(held_out=meta['name']=='held',metadata_features=[0])))
                (folder/'AUDIT_COMPLETE.json').write_text(json.dumps(dict(initialized_block=100)))
                value=i+1 if i<2 else 1e9
                raw=np.full((260,len(FEATURES)),value,dtype=np.float32);raw[100:]=1e8
                arrays=dict(raw_features=raw,feature_mask=np.ones_like(raw,bool),targets=np.full((260,len(TARGETS)),value,dtype=float),
                            target_mask=np.ones((260,len(TARGETS)),bool),block=np.arange(100,360),timestamp=times[:260],kind=np.arange(260)%4,log_index=np.zeros(260,np.int32))
                for k,v in arrays.items():np.save(folder/(k+'.npy'),v)
            finalize(SimpleNamespace(out=root),protocol,metas)
            scaler=json.loads((root/'scalers.json').read_text())
            self.assertEqual(scaler['scalers']['raw_features']['mean'][0],1.5)
            self.assertEqual(set(scaler['fit_pool_moments']['raw_features']),{'a','b'})
            with self.assertRaises(ValueError):PooledDataset(root,'held','train')
            with self.assertRaises(ValueError):PooledDataset(root,'a','train',seq_len=128)
            ds=PooledDataset(root,'a','train');self.assertEqual(len(ds),36)
            self.assertEqual(ds[0]['current_timestamp'],times[63]);self.assertEqual(ds[0]['x'].shape,(64,len(FEATURES)))
            self.assertGreater(len(PooledDataset(root,'held','heldout_test')),0)


if __name__=='__main__':unittest.main()
