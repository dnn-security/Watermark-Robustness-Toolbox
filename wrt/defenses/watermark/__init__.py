"""
Module implementing the watermarking schemes for deep neural networks.
"""
from .watermark import Watermark

from .adi import Adi
from .blackmarks import Blackmarks
from .dawn import Dawn
from .deepmarks import Deepmarks
from .deepsign import DeepSignWB
from .frontier_stitching import FrontierStitching
from .jia import Jia
from .uchida import Uchida
from .zhang import ZhangContent, ZhangUnrelated, ZhangNoise

