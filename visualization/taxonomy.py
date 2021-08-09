
# Watermarking Scheme Categories.
BLACKBOX_BACKDOOR = 'Model Independent'
WHITEBOX_BACKDOOR = 'Model Dependent'
PARAMETER_EMBEDDING = 'Parameter Encoding'
POSTPROCESSING = 'Active'

# Attack Categories.
MODEL_MODIFICATION = 'Model Modification'
MODEL_EXTRACTION = 'Model Extraction'
INPUT_PREPROCESSING = 'Input Preprocessing'

# Global colors *and ordering* for the defenses.
defense_to_color = {
     BLACKBOX_BACKDOOR: "tab:blue",
     "Content": "tab:blue",
     "Noise": "tab:orange",
     "Unrelated": "tab:green",
     "Adi": "purple",
     "Jia": "blue",
     "Li": "tab:pink",
     WHITEBOX_BACKDOOR: "tab:purple",
     "Frontier Stitching": "tab:purple",
     "Blackmarks": "k",
     PARAMETER_EMBEDDING: "tab:red",
     "Deepmarks": "gray",
     "Deepsigns": "red",
     "Uchida": "deepskyblue",
     POSTPROCESSING: "green",
     "DAWN": "green"
}

# Global colors *and ordering* for the attacks.
attack_to_color = {
    MODEL_MODIFICATION: "tab:blue",
    "FTLL": "tab:blue",
    "FTAL": "tab:blue",
    "RTLL": "tab:blue",
    "RTAL": "tab:blue",
    "Weight Pruning": "tab:blue",
    "Weight Shifting": "tab:blue",
    "Overwriting": "tab:blue",
    "Regularization": "tab:blue",
    "Neural Unlearning": "tab:blue",
    "Neural Pruning": "tab:blue",
    "Fine Pruning": "tab:blue",
    "Adversarial Training": "tab:blue",
    "Neural Laundering": "tab:blue",
    "Weight Quantization": "tab:blue",
    INPUT_PREPROCESSING: "tab:red",
    "Input Quantization": "tab:blue",
    "Input Smoothing": "tab:blue",
    "Input Noising": "tab:blue",
    "Input Reconstruction": "tab:blue",
    "Feature Squeezing": "tab:blue",
    MODEL_EXTRACTION: "tab:green",
    "Retraining": "tab:blue",
    "Knockoff Nets": "tab:blue",
    "Transfer Learning": "tab:blue",
    "Transfer Learning + FTAL": "tab:blue",
    "Distillation": "tab:blue",
    "Adversarial Model Extraction": "tab:blue"
}

# Labels used for the dataset in the plot.
dataset_labels = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "imagenet": "ImageNet"
}

# Watermarking Scheme Categories and all their members.
scheme_category_to_defense = {
    BLACKBOX_BACKDOOR: ['Content', 'Noise', 'Unrelated', 'Adi',  'Li'],
    WHITEBOX_BACKDOOR: ['Frontier Stitching', 'Blackmarks', 'Jia'],
    PARAMETER_EMBEDDING: ['Deepmarks', 'Deepsigns', 'Uchida'],
    POSTPROCESSING: ['DAWN']
}
# Attack Categories and all their members.
attack_categories = {
    MODEL_MODIFICATION: ['Regularization', 'FTLL', 'FTAL', 'RTLL', 'RTAL', 'Fine Pruning',
                         'Weight Pruning', 'Neural Unlearning', 'Adversarial Training',
                         'Neural Pruning', 'Neural Laundering', 'Overwriting', 'Weight Quantization',
                         'Weight Shifting', 'Feature Permutation', 'Weight Shifting + Input Smoothing',
                         'Label Smoothing', 'Weight Quantization'],
    MODEL_EXTRACTION: ['Retraining', 'Transfer Learning', 'Adversarial Model Extraction',
                       'Cross Architecture Retraining', 'Transfer Learning + FTAL', 'Knockoff Nets', 'Distillation',
                       'Transfer Learning + Input Smoothing', 'Smooth Retraining'],
    INPUT_PREPROCESSING: ['Input Reconstruction', 'Input Noising', 'Input Smoothing',
                          'Input Quantization', 'JPEG Compression', 'Feature Squeezing', 'Input Flipping']
}

# Lists of all defenses and attacks.
all_defenses = [item for category in scheme_category_to_defense.values() for item in category]
all_attacks = [item for category in attack_categories.values() for item in category]


def get_defense_category(defense: str, throws=True):
    """ Gets the defense category for a given defense.
    """
    for category, defenses in scheme_category_to_defense.items():
        if defense in defenses or defense == category:
            return category
    if throws:
        raise ValueError("Defense '{}' not found".format(defense))


def get_attack_category(attack:str, throws=True):
    """ Gets the attack category for a given defense.
    """
    for category, attacks in attack_categories.items():
        if attack in attacks or attack == category:
            return category
    if throws:
        raise ValueError("Attack '{}' not found".format(attack))
