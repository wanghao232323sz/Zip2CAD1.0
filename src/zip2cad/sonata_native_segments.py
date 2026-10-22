"""Explicit frozen-encoder compatibility adapter for Sonata's segment_csr calls.

Forward values use native PyTorch segment_reduce (no compiled torch-scatter).
Only nonempty, complete CSR groups used by official Sonata pooling are supported.
Do not claim max-tie backward equivalence for end-to-end backbone fine tuning.
"""
import sys,types,importlib.machinery
import torch


def segment_csr(src,indptr,out=None,reduce='sum'):
    if src.ndim!=2 or indptr.ndim!=1 or indptr.numel()<2:raise ValueError('Sonata adapter requires [N,C] and 1D complete CSR')
    lengths=indptr[1:]-indptr[:-1]
    if int(indptr[0])!=0 or int(indptr[-1])!=len(src) or bool((lengths<=0).any()):raise ValueError('Nonempty complete CSR groups required')
    if reduce not in ['sum','mean','max','min']:raise ValueError('Unsupported reduction')
    result=torch.segment_reduce(src,reduce,lengths=lengths,axis=0)
    if out is not None:out.copy_(result);return out
    return result


def install_frozen_forward_adapter():
    if 'torch_scatter' in sys.modules:raise RuntimeError('Do not replace an already imported torch_scatter')
    module=types.ModuleType('torch_scatter');module.__spec__=importlib.machinery.ModuleSpec('torch_scatter',loader=None)
    module.segment_csr=segment_csr;module.__version__='project_native_segment_reduce_frozen_forward_only'
    sys.modules['torch_scatter']=module
    return module
