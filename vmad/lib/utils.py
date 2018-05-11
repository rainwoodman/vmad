from vmad.core.operator import operator

@operator
class watchpoint:
    ain  = {'x': '*'}
    aout = {}

    def apl(self, x, monitor=print):
        monitor(x)
        return dict(y = x)

@operator
class assert_isinstance:
    ain = 'obj'
    aout = []
    def apl(self, obj, class_or_tuple):
        if not isinstance(obj, class_or_tuple):
            raise TypeError('Expecting an instance of %s, got %s', repr(class_or_tuple), repr(type(obj)))

