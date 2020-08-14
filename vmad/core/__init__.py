def get_stdlib(_stdlib=[]):
    if not _stdlib:
        from . import stdlib
        _stdlib.append(stdlib)
    return _stdlib[0]

def get_autodiff(_autodiff=[]):
    if not _autodiff:
        from . import autodiff
        _autodiff.append(autodiff)
    return _autodiff[0]
