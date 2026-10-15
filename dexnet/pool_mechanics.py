"""Orientation-independent integer v3 transitions and causal inventory support."""
import math
from dataclasses import replace
from .events import LiquidityLedger
from .tick_mechanics import (Q96,MIN_TICK,MAX_TICK,ceil_div,delta0,delta1,sqrt_at_tick,
                             tick_at_sqrt,swap_base,modify_liquidity,TickBook)


def swap_token0(state,book,amount0,fee):
    if amount0==0 or not 0<=fee<1000000:raise ValueError('Invalid swap')
    s=replace(state);left=amount0>0;remaining=abs(amount0);amount1=0;total_fee=0;crossed=[]
    for steps in range(10000):
        if not remaining:break
        nxt,initialized=book.next(s.tick,left);target=sqrt_at_tick(nxt)
        if left and target>s.sqrt or not left and target<s.sqrt:raise ValueError('Invalid tick direction')
        if left:
            net=remaining*(1000000-fee)//1000000
            boundary=delta0(target,s.sqrt,s.liquidity,True)
            new=target if net>=boundary else ceil_div((s.liquidity<<96)*s.sqrt,(s.liquidity<<96)+net*s.sqrt)
            inp=delta0(new,s.sqrt,s.liquidity,True);out=delta1(new,s.sqrt,s.liquidity,False)
            charge=ceil_div(inp*fee,1000000-fee) if new==target else remaining-inp
            remaining-=inp+charge;amount1-=out
        else:
            boundary=delta0(s.sqrt,target,s.liquidity,False)
            if remaining>=boundary:new=target
            else:
                denominator=(s.liquidity<<96)-remaining*s.sqrt
                if denominator<=0:raise ValueError('Exhausted pool range')
                new=ceil_div((s.liquidity<<96)*s.sqrt,denominator)
            out=min(remaining,delta0(s.sqrt,new,s.liquidity,False));inp=delta1(s.sqrt,new,s.liquidity,True)
            charge=ceil_div(inp*fee,1000000-fee);remaining-=out;amount1+=inp+charge
        total_fee+=charge
        if new==target:
            if initialized:
                gross,net=book.tick(nxt)
                if gross<=0:raise ValueError('Bitmap/tick inconsistency')
                s.liquidity+=-net if left else net;crossed.append(nxt)
                if s.liquidity<0:raise ValueError('Negative crossed liquidity')
            s.tick=nxt-1 if left else nxt
        elif new!=s.sqrt:s.tick=tick_at_sqrt(new)
        s.sqrt=new
        if s.sqrt in [sqrt_at_tick(MIN_TICK),sqrt_at_tick(MAX_TICK)] and remaining:raise ValueError('Exhausted pool range')
    else:raise RuntimeError('Swap step budget exceeded')
    return s,dict(amount0=amount0,amount1=amount1,fee_native=total_fee,crossed=crossed,steps=steps)


def swap_economic_base(state,book,amount,meta):
    return swap_token0(state,book,amount,meta['fee']) if meta['base_is_token0'] else swap_base(state,book,amount,meta['fee'])


class InventoryCapacity:
    """Exact output capacity, including each initialized-tick/bitmap-step floor.

    Inventory-only prefix sums are cached between LP events. Swaps then query
    one partial interval plus a prefix/suffix in logarithmic time.
    """
    def __init__(self,ranges,spacing,base0):
        from bisect import bisect_right
        self.base0=base0;self.bisect_right=bisect_right
        net={};lowest=MAX_TICK;highest=MIN_TICK
        for (lo,hi),L in ranges.items():
            net[lo]=net.get(lo,0)+L;net[hi]=net.get(hi,0)-L
            lowest=min(lowest,lo);highest=max(highest,hi)
        if not net:self.ticks=[];self.roots=[];self.prefix=[0];return
        # next(left) stops at word starts; next(right) stops at word ends.
        word=256*spacing;offset=255*spacing if base0 else 0
        cuts=set(net)
        for k in range((lowest-offset)//word,(highest-offset)//word+2):
            t=k*word+offset
            if lowest<t<highest:cuts.add(t)
        self.ticks=sorted(cuts);self.roots=[sqrt_at_tick(t) for t in self.ticks]
        self.prefix=[0];L=0
        for i,t in enumerate(self.ticks[:-1]):
            L+=net.get(t,0)
            amount=delta0(self.roots[i],self.roots[i+1],L,False) if base0 else delta1(self.roots[i],self.roots[i+1],L,False)
            self.prefix.append(self.prefix[-1]+amount)
    def query(self,state):
        if not self.ticks:return 0
        if self.base0:
            i=self.bisect_right(self.ticks,state.tick)
            if i==len(self.ticks):return 0
            return delta0(state.sqrt,self.roots[i],state.liquidity,False)+self.prefix[-1]-self.prefix[i]
        i=self.bisect_right(self.ticks,state.tick)-1
        if i<0:return 0
        return delta1(self.roots[i],state.sqrt,state.liquidity,False)+self.prefix[i]


class CapacityLedger(LiquidityLedger):
    def __init__(self,spacing):
        super().__init__(spacing);self.capacity_cache={}
    def apply(self,r):
        super().apply(r)
        if r['event'] in ['Mint','Burn'] and r['amount']:self.capacity_cache.clear()
    def capacity(self,base0):
        from .tick_mechanics import PoolState
        if base0 not in self.capacity_cache:self.capacity_cache[base0]=InventoryCapacity(self.ranges,self.spacing,base0)
        return self.capacity_cache[base0].query(PoolState(self.sqrt,self.tick,self.L))
    def log_reserve(self,base0):
        if self.L<=0:raise ValueError('No active liquidity for relative amount')
        return math.log(self.L)+(math.log(Q96)-math.log(self.sqrt) if base0 else math.log(self.sqrt)-math.log(Q96))
    def book(self):
        book=TickBook(self.spacing,lambda word:0,lambda tick:(0,0))
        for (lo,hi),L in self.ranges.items():book.change(lo,L,L);book.change(hi,L,-L)
        return book


def canonical_inventory(ledger,base0):
    inv={}
    for (lo,hi),value in ledger.ranges.items():
        lower,upper=(lo//ledger.spacing,hi//ledger.spacing) if base0 else (-hi//ledger.spacing,-lo//ledger.spacing)
        inv.setdefault(lower,{})[upper]=value
    return inv
