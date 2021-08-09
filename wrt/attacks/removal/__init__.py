"""
Module providing removal attacks under a common interface.
Removal attack have white-box access to the target model and remove the watermark within the target model.
"""
from .fine_tuning import FTLLAttack, FTALAttack, RTLLAttack, RTALAttack
from .fine_pruning import FinePruningAttack
from .transfer_learning import TransferLearningAttack
from .neural_cleanse import NeuralCleanseUnlearning, NeuralCleansePartialUnlearning, NeuralCleanseMultiUnlearning, \
    NeuralCleansePruning, NeuralCleansePartialPruning, NeuralCleanseMultiPruning
from .regularization import Regularization
from .laundering import Laundering, PartialLaundering, MultiLaundering
from .adversarial_training import AdversarialTraining
from .weight_pruning import WeightPruning
from .input_reconstruction import InputReconstruction
from .input_noising import InputNoising
from .input_horizontal_flipping import InputHorizontalFlipping
from .input_smoothing import InputMeanSmoothing, InputGaussianSmoothing, InputMedianSmoothing
from .input_quantization import InputQuantization
from .jpeg_compression import JPEGCompression
from .feature_squeezing import FeatureSqueezing
from .knockoff_nets import KnockoffNets
from .feature_shuffling import FeatureShuffling
from .weight_quantization import WeightQuantization
from .weight_shifting import WeightShifting
from .label_smoothing import LabelSmoothingAttack
from .ensemble import EnsembleAttack
from .distillation import ModelDistillation
from .random_occlusion import RandomOcclusion
from .label_noising import LabelNoisingAttack
from .feature_regularization import FeatureRegularization
from .retraining import ModelExtraction
