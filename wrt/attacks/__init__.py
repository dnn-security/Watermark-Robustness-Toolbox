"""
Module providing attacks on neural network watermarking under a common interface.
"""
from wrt.attacks.attack import Attack, RemovalAttack, ExtractionAttack

from wrt.attacks.removal.fine_tuning import FTLLAttack, FTALAttack, RTLLAttack, RTALAttack, FineTuningAttack
from wrt.attacks.removal.fine_pruning import FinePruningAttack
from wrt.attacks.removal.transfer_learning import TransferLearningAttack
from wrt.attacks.removal.neural_cleanse import NeuralCleanseUnlearning, NeuralCleansePartialUnlearning, NeuralCleanseMultiUnlearning, \
    NeuralCleansePruning, NeuralCleansePartialPruning, NeuralCleanseMultiPruning
from wrt.attacks.removal.regularization import Regularization
from wrt.attacks.removal.stacked_attack import StackedAttack
from wrt.attacks.removal.laundering import Laundering, PartialLaundering, MultiLaundering
from wrt.attacks.removal.adversarial_training import AdversarialTraining
from wrt.attacks.removal.weight_pruning import WeightPruning
from wrt.attacks.removal.input_reconstruction import InputReconstruction
from wrt.attacks.removal.input_noising import InputNoising
from wrt.attacks.removal.input_smoothing import InputMeanSmoothing, InputGaussianSmoothing, InputMedianSmoothing
from wrt.attacks.removal.input_quantization import InputQuantization
from wrt.attacks.removal.jpeg_compression import JPEGCompression
from wrt.attacks.removal.feature_squeezing import FeatureSqueezing
from wrt.attacks.removal.knockoff_nets import KnockoffNets

from wrt.attacks.extraction.neural_cleanse import NeuralCleanse, NeuralCleansePartial, NeuralCleanseMulti
