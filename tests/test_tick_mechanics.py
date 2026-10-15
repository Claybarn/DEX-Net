import unittest
from dataclasses import replace
from dexnet.tick_mechanics import (Q96,MIN_TICK,MAX_TICK,sqrt_at_tick,tick_at_sqrt,
    PoolState,TickBook,swap_base,delta1,modify_liquidity,constant_base_sqrt,ceil_div)


def book_from_ticks(ticks,spacing=10):
    def word(w):
        return sum(1<<(t//spacing%256) for t,(gross,net) in ticks.items() if gross and t//spacing//256==w)
    return TickBook(spacing,word,lambda t:ticks.get(t,(0,0)))


class TickMechanicsTests(unittest.TestCase):
    def test_tick_ratios_match_contract_golden_values_and_inverse(self):
        for tick,value in [(MIN_TICK,4295128739),(MAX_TICK,1461446703485210103287273052203988822378723970342),
                           (0,Q96),(1,79232123823359799118286999568),(-1,79224201403219477170569942574)]:
            self.assertEqual(sqrt_at_tick(tick),value)
        for tick in [-887000,-2561,-10,-1,0,1,100,222000,887000]:
            self.assertEqual(tick_at_sqrt(sqrt_at_tick(tick)),tick)
            self.assertEqual(tick_at_sqrt(sqrt_at_tick(tick)-1),tick-1)

    def test_bitmap_word_boundaries_and_negative_floor(self):
        b=book_from_ticks({-2570:(1,1),-10:(1,1),0:(1,1),2560:(1,1)})
        self.assertEqual(b.next(-1,True),(-10,True))
        self.assertEqual(b.next(-1,False),(0,True))
        self.assertEqual(b.next(0,False),(2550,False))
        self.assertEqual(b.next(2550,False),(2560,True))

    def test_no_crossing_matches_constant_liquidity_with_fee_in_both_directions(self):
        state=PoolState(Q96,0,10**18)
        for amount in [10**9,-10**9,1]:
            b=book_from_ticks({});pred,d=swap_base(state,b,amount)
            self.assertEqual(pred.sqrt,constant_base_sqrt(state.sqrt,state.liquidity,amount))
            self.assertEqual(d['crossed'],[]);self.assertEqual(state.sqrt,Q96)
            if amount>0:self.assertLessEqual(d['amount0'],0)
            else:self.assertGreater(d['amount0'],0)

    def test_crossing_updates_liquidity_and_price_in_each_direction(self):
        L=10**18;ticks={-1000:(L,L),-100:(L,L),100:(L,-L),1000:(L,-L)}
        for sign in [-1,1]:
            state=PoolState(Q96,0,2*L);boundary=sqrt_at_tick(sign*100)
            first=delta1(Q96,boundary,2*L,sign>0);remaining=10**14
            pred,d=swap_base(state,book_from_ticks(ticks),sign*(first+remaining),fee=0)
            expected=boundary+(remaining*Q96//L if sign>0 else -ceil_div(remaining*Q96,L))
            self.assertEqual(pred.sqrt,expected);self.assertEqual(pred.liquidity,L)
            self.assertEqual(d['crossed'],[sign*100])

    def test_mint_burn_and_frozen_snapshot_have_correct_temporal_order(self):
        L=10**18;ticks={-100:(L,L),100:(L,-L)};state=PoolState(Q96,0,L)
        b=book_from_ticks(ticks);snapshot=b.fork();prior=replace(state)
        modify_liquidity(state,b,-100,100,L)
        self.assertEqual(state.liquidity,2*L);self.assertEqual(b.tick(100),(2*L,-2*L))
        self.assertEqual(snapshot.tick(100),(L,-L))
        new,_=swap_base(state,b,10**9);old,_=swap_base(prior,snapshot,10**9)
        self.assertLess(new.sqrt,old.sqrt)
        modify_liquidity(state,b,-100,100,-2*L)
        self.assertEqual(state.liquidity,0);self.assertEqual(b.tick(100),(0,0))
        self.assertFalse(b.word(0)&(1<<10))


if __name__=='__main__':unittest.main()
