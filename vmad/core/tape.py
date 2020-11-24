from . import get_autodiff
import numpy
import resource
import os
from pathlib import Path
import psutil


class Record(object):
    """ A record on the tape. 
        A record contains the node and the resolved arg symbols to the node.

    """
    def __init__(self, node, impl_kwargs):
        self.node = node
        self.impl_kwargs = impl_kwargs

    def __repr__(self):
        return '%s / %s' % (self.node, self.impl_kwargs)

class Tape(list):

    tape_num = 0
    rss      = [0]
    num_finalized = 0

    def __init__(self, model, init):
        self.model       = model
        self.init        = init
        self._completed  = False
        self.num         = Tape.tape_num
        if self.num==0:
            Tape.rss[0]=self.get_current_mem_usage()
        Tape.tape_num+=1 

    def finalize(self, out):
        """ Finalize the tape, with a set of computed outputs.

            Parameters
            ----------
            out : dict / OrderedDict
        """
        assert isinstance(out, dict)
        self.out = out.copy()
        self._completed = True
        Tape.num_finalized+=1
        if Tape.num_finalized==Tape.tape_num:
            self.dump_mem_usage()

    def append(self, node, impl_kwargs):
        assert not self._completed
        list.append(self, Record(node, impl_kwargs))
        Tape.rss.append(self.get_current_mem_usage())        

    def get_vjp_vout(self):
        return ['_' + varname for varname in self.init.keys()]

    def get_jvp_vout(self):
        return [varname + '_' for varname in self.out.keys()]

    def get_vjp(self):
        assert self._completed
        return get_autodiff().vjpmodel(self)

    def get_jvp(self):
        assert self._completed
        return get_autodiff().jvpmodel(self)

    def compute_jvjp(self, vout, aout, init):
        jvp = self.get_jvp()
        aout_ = [a + '_' for a in aout]
        t = jvp.compute(aout_, init)
        vjp = self.get_vjp()
        p = vjp.compute(vout, init=dict([('_' + a, t1) for a, t1 in zip(aout, t)]))
        return p

    def dump_mem_usage(self):
        numpy.save(os.path.join(os.getcwd(),"mem_profiles/mem_%d.npy"%self.num),Tape.rss)

    def get_current_mem_usage(self):
        process = psutil.Process(os.getpid())
        rss     = process.memory_info().rss
        return rss/1e6

