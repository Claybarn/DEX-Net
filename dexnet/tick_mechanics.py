"""Integer tick-aware mechanics for USDC(token0)/WETH(token1) replay.

The swap is specified in token1 units: exact input when selling WETH, exact
output when buying WETH. This matches the model's base-size convention. Integer
rounding follows Uniswap v3's published mathematical rules; no price outcome is
used as a stopping limit. Tick ratios use independently generated Q128 factors.
References: Uniswap/v3-core TickMath, SqrtPriceMath, SwapMath, TickBitmap.
"""
from dataclasses import dataclass,replace
from decimal import Decimal,localcontext,ROUND_HALF_UP
from functools import lru_cache

Q96=1<<96
MIN_TICK=-887272
MAX_TICK=887272


def _factors():
    with localcontext() as ctx:
        ctx.prec=150
        return tuple(int((Decimal(2)**128/Decimal('1.0001')**(Decimal(2**i)/2)).to_integral_value(rounding=ROUND_HALF_UP)) for i in range(20))


FACTORS=_factors()


def ceil_div(a,b):
    if a<0 or b<=0:raise ValueError('Nonnegative numerator and positive denominator required')
    return (a+b-1)//b


@lru_cache(maxsize=65536)
def sqrt_at_tick(tick):
    if not MIN_TICK<=tick<=MAX_TICK:raise ValueError('Tick out of bounds')
    ratio=1<<128
    for i,factor in enumerate(FACTORS):
        if abs(tick)&(1<<i):ratio=(ratio*factor)>>128
    if tick>0:ratio=((1<<256)-1)//ratio
    return ceil_div(ratio,1<<32)


def tick_at_sqrt(sqrt):
    if not sqrt_at_tick(MIN_TICK)<=sqrt<sqrt_at_tick(MAX_TICK):raise ValueError('Price out of bounds')
    lo,hi=MIN_TICK,MAX_TICK
    while lo+1<hi:
        mid=(lo+hi)//2
        if sqrt_at_tick(mid)<=sqrt:lo=mid
        else:hi=mid
    return lo


def delta0(a,b,liquidity,up):
    a,b=sorted([a,b]);num=(liquidity<<96)*(b-a);den=a*b
    return ceil_div(num,den) if up else num//den


def delta1(a,b,liquidity,up):
    num=liquidity*abs(b-a)
    return ceil_div(num,Q96) if up else num//Q96


@dataclass
class PoolState:
    sqrt:int
    tick:int
    liquidity:int


class TickBook:
    """Lazy historical base plus event-ordered local updates; fork is a snapshot."""
    def __init__(self,spacing,read_word,read_tick):
        self.spacing=spacing;self.read_word=read_word;self.read_tick=read_tick
        self.words={};self.ticks={}

    def fork(self):
        other=TickBook(self.spacing,self.read_word,self.read_tick)
        other.words=self.words.copy();other.ticks=self.ticks.copy()
        return other

    def word(self,word):
        if word not in self.words:self.words[word]=self.read_word(word)
        return self.words[word]

    def tick(self,tick):
        if tick not in self.ticks:self.ticks[tick]=tuple(self.read_tick(tick))
        return self.ticks[tick]  # gross, net

    def change(self,tick,gross_delta,net_delta):
        if tick%self.spacing:raise ValueError('Unaligned initialized tick')
        gross,net=self.tick(tick);gross+=gross_delta;net+=net_delta
        if gross<0:raise ValueError('Burn exceeds reconstructed liquidity')
        if not gross and net:raise ValueError('Empty tick with nonzero net')
        compressed=tick//self.spacing;word,bit=divmod(compressed,256)
        bits=self.word(word)
        self.words[word]=(bits|(1<<bit)) if gross else (bits&~(1<<bit))
        self.ticks[tick]=(gross,net)

    def next(self,tick,left):
        compressed=tick//self.spacing  # Python floors negative ticks as required.
        if left:
            word,bit=divmod(compressed,256);bits=self.word(word)&((1<<(bit+1))-1)
            index=word*256+(bits.bit_length()-1 if bits else 0)
        else:
            word,bit=divmod(compressed+1,256);bits=self.word(word)&(((1<<256)-1)^((1<<bit)-1))
            index=word*256+((bits&-bits).bit_length()-1 if bits else 255)
        return max(MIN_TICK,min(MAX_TICK,index*self.spacing)),bool(bits)


def modify_liquidity(state,book,lower,upper,amount):
    if lower>=upper:raise ValueError('Invalid liquidity range')
    if amount==0:return
    book.change(lower,amount,amount);book.change(upper,amount,-amount)
    if lower<=state.tick<upper:state.liquidity+=amount
    if state.liquidity<0:raise ValueError('Negative active liquidity')


def swap_base(state,book,amount1,fee=500):
    """Return a predicted post-state and native amounts; inputs are not mutated."""
    if amount1==0 or not 0<=fee<1_000_000:raise ValueError('Invalid swap')
    s=replace(state);left=amount1<0;remaining=abs(amount1);amount0=0;total_fee=0;crossed=[]
    for steps in range(10000):
        if not remaining:break
        nxt,initialized=book.next(s.tick,left);target=sqrt_at_tick(nxt)
        if left and target>s.sqrt or not left and target<s.sqrt:raise ValueError('Invalid tick direction')
        # Token1 output (left) or input after fee (right) determines sqrt price.
        if left:
            to_boundary=delta1(target,s.sqrt,s.liquidity,False)
            new=target if remaining>=to_boundary else s.sqrt-ceil_div(remaining*Q96,s.liquidity)
            out=min(remaining,delta1(new,s.sqrt,s.liquidity,False))
            inp=delta0(new,s.sqrt,s.liquidity,True)
            charge=ceil_div(inp*fee,1_000_000-fee)
            remaining-=out;amount0+=inp+charge
        else:
            net=remaining*(1_000_000-fee)//1_000_000
            to_boundary=delta1(s.sqrt,target,s.liquidity,True)
            new=target if net>=to_boundary else s.sqrt+net*Q96//s.liquidity
            inp=delta1(s.sqrt,new,s.liquidity,True);out=delta0(s.sqrt,new,s.liquidity,False)
            charge=ceil_div(inp*fee,1_000_000-fee) if new==target else remaining-inp
            remaining-=inp+charge;amount0-=out
        total_fee+=charge
        if new==target:
            if initialized:
                gross,liquidity_net=book.tick(nxt)
                if gross<=0:raise ValueError('Bitmap/tick inconsistency')
                s.liquidity+=(-liquidity_net if left else liquidity_net);crossed.append(nxt)
                if s.liquidity<0:raise ValueError('Negative crossed liquidity')
            s.tick=nxt-1 if left else nxt
        elif new!=s.sqrt:s.tick=tick_at_sqrt(new)
        s.sqrt=new
        if s.sqrt in [sqrt_at_tick(MIN_TICK),sqrt_at_tick(MAX_TICK)] and remaining:raise ValueError('Exhausted pool range')
    else:raise RuntimeError('Swap step budget exceeded')
    return s,dict(amount0=amount0,amount1=amount1,fee_native=total_fee,crossed=crossed,steps=steps)


def constant_base_sqrt(sqrt,liquidity,amount1,fee=500):
    if liquidity<=0:return None
    if amount1>0:return sqrt+(amount1*(1_000_000-fee)//1_000_000)*Q96//liquidity
    result=sqrt-ceil_div(-amount1*Q96,liquidity)
    return result if result>0 else None
