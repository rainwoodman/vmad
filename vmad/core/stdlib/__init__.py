from vmad.core.operator import operator

# special operator for marking an output
@operator
class terminal:
    ain  = 'x'
    aout = 'y'

    def apl(self, x):
        return dict(y=x)
    def vjp(self, _y):
        return dict(_x=_y)
    def jvp(self, x_):
        return dict(y_=x_)


from .utils import eval, watchpoint
from .utils import assert_isinstance, assert_true

from .operators import pos, neg, add, sub, mul, div, pow, abs

