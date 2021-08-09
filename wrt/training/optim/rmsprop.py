''' https://github.com/narumiruna/efficientnet-pytorch/blob/master/efficientnet/optim/rmsprop.py '''

import mlconfig
import torch
from torch import optim


@mlconfig.register
class TFRMSprop(optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=1e-5, rho=0.9, eps=1e-3, momentum=0.9, warmup=0):
        """
        https://github.com/tensorflow/tpu/blob/cab34d82a974ca4f8ced19c236462b446f0feadf/models/official/efficientnet/utils.py#L65
        https://github.com/tensorflow/tensorflow/blob/97fb325e3b8499d375251359fd69abd2fa96ee39/tensorflow/python/training/rmsprop.py#L54
        https://github.com/tensorflow/tensorflow/blob/d5c6687d9919c562bea2a01a6e1be1756bfaab33/tensorflow/core/kernels/training_ops.cc#L453
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, rho=rho, eps=eps, momentum=momentum, warmup=warmup)
        super(TFRMSprop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['ms'] = torch.ones_like(p)
                    state['mom'] = torch.zeros_like(p)

                # weight decay
                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                state['step'] += 1

                warmup = group['warmup']
                step = state['step']
                lr = group['lr']
                if warmup > step:
                    lr *= step / warmup

                rho = group['rho']
                ms = state['ms']
                ms.add_(1 - rho, grad.pow(2) - ms)
                mom = state['mom']
                mom.mul_(group['momentum']).addcdiv_(lr, grad, ms.add(group['eps']).sqrt())

                p.data.add_(-mom)

        return loss