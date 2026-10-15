import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_model import SCALERS,batch
from dexnet.model import PooledEventModel
from dexnet.coverage import make_schedule,CoverageSchedule
from scripts.train_round import attach_ids


class FullPooledTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_balanced_coverage_reproducible_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';sizes={'large':19,'small':3}
            spec=make_schedule(a,sizes,epochs=3,batch_size=4,seed=7);make_schedule(b,sizes,epochs=3,batch_size=4,seed=7)
            left=CoverageSchedule(a);right=CoverageSchedule(b);epoch_batches=len(left)//3
            for epoch in range(3):
                seen={n:[] for n in sizes}
                for i in range(epoch*epoch_batches,(epoch+1)*epoch_batches):
                    n,rows=left[i];seen[n].extend(rows);other,copy=right[i];self.assertEqual(n,other);np.testing.assert_array_equal(rows,copy)
                self.assertEqual(len(seen['large']),len(seen['small']))
                for n in sizes:
                    self.assertEqual(set(seen[n]),set(range(sizes[n])))
                    self.assertEqual(len(set(seen[n][:sizes[n]])),sizes[n])
            # Restarting at a checkpoint retrieves the same remaining schedule.
            again=CoverageSchedule(a)
            for i in range(7,len(left)):
                self.assertEqual(left[i][0],again[i][0]);np.testing.assert_array_equal(left[i][1],again[i][1])
    def test_embedding_unknown_is_neutral_and_never_fitted(self):
        torch.manual_seed(11);base=PooledEventModel(SCALERS).eval()
        embedded=PooledEventModel(SCALERS,pool_count=8,pool_dropout=0).eval()
        missing=embedded.load_state_dict(base.state_dict(),strict=False).missing_keys
        self.assertEqual(missing,['pool_embedding.weight']);b=batch();b['pool_id']=torch.tensor([1,2,0,1])
        with torch.no_grad():embedded.pool_embedding.weight[1:]=.1*torch.randn_like(embedded.pool_embedding.weight[1:])
        unknown=embedded.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'],torch.zeros(4,dtype=torch.long))
        expected=base.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'])
        torch.testing.assert_close(unknown,expected,rtol=0,atol=0)
        loss,_=embedded.losses(b);loss.backward()
        self.assertEqual(float(embedded.pool_embedding.weight.grad[0].abs().sum()),0)
        self.assertGreater(float(embedded.pool_embedding.weight.grad[1:3].abs().sum()),0)
        self.assertEqual(float(embedded.pool_embedding.weight.grad[3:].abs().sum()),0)
        opt=torch.optim.AdamW(embedded.parameters(),lr=.001);opt.step()
        self.assertEqual(float(embedded.pool_embedding.weight[0].abs().sum()),0)
        datasets={'seen':SimpleNamespace(),'unseen':SimpleNamespace()};attach_ids(datasets,{'pool_ids':{'seen':1}})
        self.assertEqual(datasets['seen'].pool_id,1);self.assertEqual(datasets['unseen'].pool_id,0)
    def test_larger_model_all_heads_have_finite_gradients(self):
        m=PooledEventModel(SCALERS,d_model=128,layers=4);loss,_=m.losses(batch());loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_embedding_dropout_uses_neutral_identity(self):
        m=PooledEventModel(SCALERS,pool_count=8,pool_dropout=1.).train();b=batch()
        with torch.no_grad():m.pool_embedding.weight[1:]=100
        torch.manual_seed(2);a=m.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'],torch.ones(4,dtype=torch.long))
        torch.manual_seed(2);z=m.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'],torch.zeros(4,dtype=torch.long))
        torch.testing.assert_close(a,z,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
