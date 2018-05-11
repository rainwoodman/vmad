class ModelError(Exception): pass

class BadArgument(ModelError): pass
class UnpackError(ModelError): pass
class DuplicatedOutput(ModelError): pass
class MissingArgument(ModelError): pass
class OverwritePrecaution(ModelError): pass
class UnexpectedOutput(ModelError): pass
class ResolveError(ModelError): pass
class InferError(ModelError): pass
class BrokenPrimitive(ModelError): pass

class ExecutionError(ModelError):
    def __init__(self, msg, reason):
        self.reason = reason
        ModelError.__init__(self, msg)

def makeExecutionError(msg, reason):
    errortype = type("ExecutionError(%s)" % type(reason).__name__,
            (type(reason), ExecutionError), {})
    return errortype(msg, reason)
