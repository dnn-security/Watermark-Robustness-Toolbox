import mlconfig
from torch import optim

from .rmsprop import TFRMSprop

mlconfig.register(optim.SGD)
mlconfig.register(optim.Adam)

mlconfig.register(optim.lr_scheduler.MultiStepLR)
mlconfig.register(optim.lr_scheduler.StepLR)
mlconfig.register(optim.lr_scheduler.ExponentialLR)
mlconfig.register(optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(optim.lr_scheduler.ReduceLROnPlateau)
