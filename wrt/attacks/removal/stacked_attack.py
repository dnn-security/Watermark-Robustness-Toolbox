from abc import ABC
from copy import deepcopy
from typing import List

import mlconfig
import numpy as np
from tqdm import tqdm

from steal import __load_model
from wrt.attacks import RemovalAttack
from wrt.defenses import Watermark
import torch


class StackedAttack(RemovalAttack, ABC):
    """ Allows stacking together multiple attacks.
    """

    @staticmethod
    def __group_prefixes(attack_config: dict) -> List[dict]:
        """ Creates a dict of dict by grouping keys with the same prefix. """
        attack_configs = []
        for prefix, attack in zip(attack_config["attack_prefixes"], attack_config["attacks"]):
            attack_config_filtered = {"attack": attack}
            for key in attack_config.keys():
                if key.startswith(prefix + "_"):
                    attack_config_filtered.setdefault(key[len(prefix) + 1:], attack_config[key])
            attack_configs.append(attack_config_filtered)
        return attack_configs

    def __init__(self, source_classifier, **attack_config: dict):
        """ attack_config has to be as follows.
            init_kwargs = {
                "attacks": [FTALAttack, FTLLAttack], # Ordered
                "attack_prefixes": ["ftal", "ftll"], # Ordered
                "ftal_lr": 0.2, <- Will be sent to FTALAttack without prefix
                "ftll_lr": 0.1, <- Will be sent to FTLLAttack without prefix
            }
            """
        super().__init__(source_classifier)
        self.classifier = source_classifier
        self.attack_init_kwargs = self.__group_prefixes(attack_config)

    def get_classifier(self):
        return self.classifier

    def remove(self, x, y=None, wm_data=None, val_data=None, **kwargs):
        """ Initializes an attack, executes the removal, stores the surrogate model
            and passes it on to the next method.
        """
        attack_remove_kwargs = self.__group_prefixes(kwargs)

        removal_identifier = str(np.random.randint(1000000)).zfill(6)
        with tqdm(zip(self.attack_init_kwargs, attack_remove_kwargs), disable=True) as pbar:
            previous_attack = self
            for i, (init_kwargs, remove_kwargs) in enumerate(pbar):
                attack_ref, attack_ref2 = init_kwargs.pop("attack", None), remove_kwargs.pop("attack", None)
                pbar.set_description(str(attack_ref))
                print(f"Running {attack_ref} ..")

                 # Run the attack.
                attack = attack_ref(classifier=previous_attack.get_classifier(), **init_kwargs)
                attack.remove(x=x, y=y, valid_loader=val_data, wm_data=wm_data, **remove_kwargs)
                previous_attack = attack

                self.classifier = attack.get_classifier()

                classifier_savepath = removal_identifier + "_" + str(i) + "_stacked_attack.pth"
                print(f"Saving classifier at '{classifier_savepath}'")
                try:
                    self.classifier.save(classifier_savepath)
                except Exception as e:
                    print(e)
        print("Done removing!")


@mlconfig.register
def stacked_attack(**kwargs):
    return None


@mlconfig.register
def stacked_attack_removal(source_model, config, wm_data, train_loader, valid_loader, output_dir=None,  **kwargs):
    attack_keys = config.attack_list

    attack = None
    for i, attack in enumerate(attack_keys):
        sub_config = config[attack]
        print("###########", attack, sub_config)

        if "surrogate_model" in sub_config.keys():
            surrogate_model = sub_config.surrogate_model()
            optimizer = sub_config.optimizer(surrogate_model.parameters())
            surrogate_model = __load_model(surrogate_model, optimizer,
                                           image_size=sub_config.surrogate_model.image_size,
                                           num_classes=sub_config.surrogate_model.num_classes)
        else:
            surrogate_model = deepcopy(source_model)

        if sub_config.setdefault("predict_labels", False):
            print("Predicting new labels!")
            if sub_config.setdefault("true_labels", False):
                train_loader = sub_config.dataset(train=True)
            else:
                train_loader = sub_config.dataset(source_model=source_model, train=True)

        attack: RemovalAttack = sub_config.create(classifier=surrogate_model, config=sub_config)

        defense: Watermark = wm_data[0]
        # Choose the correct pretrained model for the defense.
        if i == 0 and "pretrained" in sub_config.keys() and defense.get_name().lower() in sub_config.pretrained.keys():
            path_to_checkpoint = sub_config.pretrained[defense.get_name().lower()]
            print(f"loading a pretrained model from '{path_to_checkpoint}'!")

            checkpoint = torch.load(path_to_checkpoint)
            attack.get_classifier().model.load_state_dict(checkpoint["model"])
            attack.get_classifier().optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            # Check if there is a pretrained model.
            attack, train_metric = sub_config.remove(attack=attack,
                                                     source_model=source_model,
                                                     train_loader=train_loader,
                                                     valid_loader=valid_loader,
                                                     config=sub_config,
                                                     output_dir=output_dir,
                                                     wm_data=wm_data)
        source_model = attack.get_classifier()
        print(attack)
    return attack, {}