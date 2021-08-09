""" Takes a tabular .csv file as input and parses it into a pandas table.
Provides all convenience functions, such as computing the Nash Equilibrium for a set of attacks/defenses.

Reads all data that is available to it.

If you want to add a data file (and a function to parse it, you can do that here)
"""
import abc
from typing import List

import pandas as pd
import numpy as np

from pandas import DataFrame, Series

# Paths to csv.
config = {
    "cifar": {
        "attack_data": "../data/experiment_results/cifar_data.csv",
        "attack_times": "../data/experiment_results/cifar_attack_runtimes.csv",
        "embed_times": "../data/experiment_results/cifar_runtime_defense.csv",
        "desc": "'attack_data' stores the watermark and test accuracies for (i) the marked source model, (ii) the unmarked"
                "null model and  (iii) the surrogate models for all attacks."
                "It also stores the 'attack_times' in seconds for all attacks."
                "and the 'embed_times' in seconds for each defense."
    },
    "imagenet": {
        "attack_data": "../data/experiment_results/imagenet_data.csv",
        "attack_times": "../data/experiment_results/imagenet_attack_runtimes.csv",
        "embed_times": "../data/experiment_results/imagenet_runtime_defense.csv",
        "desc": "'attack_data' stores the watermark and test accuracies for (i) the marked source model, (ii) the unmarked"
                "null model and  (iii) the surrogate models for all attacks."
                "'attack_times' stores the runtimes in seconds for all attacks."
                "'embed_times' stores the runtimes for the embeddings for each defense."
    }
}


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def get(dataset):
        return {
            "cifar10": CIFAR10Parser(),
            "imagenet": ImagenetNewParser()
        }[dataset.lower()]

    @staticmethod
    def equals(str1, str2):
        return (str1 in str2) and (abs(len(str1) - len(str2)) < 5)

    @abc.abstractmethod
    def get_attack_list(self):
        """ Returns a list of all attacks featured in the attack_data table
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_defense_list(self):
        """ Returns a list of all defenses featured in the attack_data table
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_defense_data(self, defense_str: str):
        """ Returns a list of data containing the watermark accuracy and test accuracy for the
        marked and unmarked model. Returns one element for each defense param config.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def slice(self, attacks: List[str], defenses: List[str]):
        """ Returns a submatrix with only the values for the attacks and defenses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_equilibrium(self, attacks: List[str], defenses: List[str]):
        """ Returns the equilibrium attack and params for the attack and defense lists
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding_times(self):
        """ Returns a dict with defenses and their embedding times in second.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_attack_times(self):
        """ Returns a dict with attacks and their runtimes in second.
        """
        raise NotImplementedError


class CIFAR10Parser(DataParser):
    def __init__(self, base_value=0.5, rescale=True):
        """ Reads all files associated with a dataset into a pandas frame.
        """
        print("Loading the CIFAR-10 parser!")
        super().__init__()
        this_config = config["cifar"]
        self.attack_data = pd.read_csv(this_config["attack_data"])
        self.embed_times = pd.read_csv(this_config["embed_times"])
        self.attack_times = pd.read_csv(this_config["attack_times"])
        self.base_value = base_value

        if rescale:
            self.attack_data = self.__rescale_wm_accs_in_frame()
            try:
                filename = "../data/experiment_results/_cifar_detailed_scaled.csv"
                self.attack_data.to_csv(filename)
                print(f"Made a copy of the scaled csv file to '{filename}'")
            except:
                pass

    def get_attack_list(self):
        """ Returns a list of all attacks featured in the attack_data table
        """
        return [x.strip() for x in self.attack_data.attacks.dropna().unique()]

    def normalize_wm_acc(self, x, decision_threshold):
        """ Normalizes the watermark accuracy given to a base given its decision threshold
        """
        return np.clip((1 - self.base_value) / (1 - decision_threshold) * x + (self.base_value - decision_threshold) / (
                1 - decision_threshold), 0, np.inf)

    def __rescale_wm_accs_in_frame(self):
        """ Applies the rescaling for each watermarking scheme column-wise
        """
        defense_data = self.get_defense_list(duplicates=True)

        for column in defense_data:
            decision_threshold = np.ceil(100*float(self.attack_data[column][1]))/100

            for i, value in enumerate(self.attack_data[column][4:].copy()):
                try:
                    wm_acc, test_acc, time = value.split("/")
                except:
                    print(f"Could not split value {value} in column {column}")
                    print(self.attack_data[column][4:])
                    exit()
                scaled_wm_acc = self.normalize_wm_acc(float(wm_acc), decision_threshold)
                self.attack_data[column][4 + i] = f"{scaled_wm_acc:.4f}/{test_acc}/{time}"

        return self.attack_data

    def slice(self, attacks: List[str], defenses: List[str]):
        """ Returns the equilibrium attack and params for the attack and defense lists
        """
        # Read the column names to perform the projection.
        column_names = []
        for defense in defenses:
            for column in self.get_defense_list(duplicates=True):
                if self.equals(defense, column):
                    column_names.append(column)

        # Get the indices of the rows that we are interested in.
        row_names = []
        for index, row in self.attack_data.iterrows():
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    row_names.append(index)

        return self.attack_data[column_names].iloc[row_names]

    def get_defense_list(self, duplicates=False):
        defenses = list(self.attack_data.columns[2:].unique().values)
        # Only take those that are not "double" indexed.
        if duplicates:
            res = defenses
        else:
            # Remove duplicates
            res = []
            for defense in defenses:
                if "." not in defense:
                    res.append(defense.strip())
        return res

    def get_defense_data(self, defense_str):
        """ Returns a list of data containing the watermark accuracy and test accuracy for the
        marked and unmarked model. Returns one element for each defense param config.
        """
        if not defense_str in self.get_defense_list():
            print(f"[ERROR] Could not find {defense_str} in data table.")
            return []

        defense_data = []
        # All columns. We need to iterate to obtain the secondary column indices.
        defenses = self.get_defense_list(duplicates=True)
        for defense in defenses:
            if self.equals(defense_str, defense):
                defense_data.append(
                    {
                        "decision_threshold": self.attack_data[defense][1],
                        "unmarked_wm_acc": self.attack_data[defense][2].split("/")[0],
                        "unmarked_test_acc": self.attack_data[defense][2].split("/")[1],
                        "marked_wm_acc": self.attack_data[defense][3].split("/")[0],
                        "marked_test_acc": self.attack_data[defense][3].split("/")[1]
                    }
                )
        return defense_data

    def compute_payoff_matrix(self, df: DataFrame):
        """ Computes the payoff matrix for computing the equilibrium.
        We set the payoff to zero for attacks whose stealing loss is greater than 5 percentage points.

        High payoff => attacker wins
        """
        # Cut-off for a successful attack w.r.t. the watermark accuracy
        decision_threshold = self.base_value
        # Cut-off for a successful attack w.r.t. the test accuracy (difference to source model acc)
        max_stealing_loss = 0.05

        # Match defenses of given matrix with those in attack_data.
        # Markup the given matrix with the source model's test accuracy
        all_defenses = list(df.columns.unique().values)
        df_copy = df.copy()
        for defense in all_defenses:
            marked_test_acc = self.attack_data[defense][2].split("/")[1]
            if type(df_copy[defense]) is Series:
                df_copy[defense] = df_copy[defense].map(lambda x: str(x) + "/" + marked_test_acc)
            else:
                df_copy[defense] = df_copy[defense].applymap(lambda x: str(x) + "/" + marked_test_acc)

        # Compute the payoff matrix
        def compute_payoff(x):
            """ Select lowest watermark retention, but if watermark retention is lower than decision threshold
            we select the highest accuracy. High payoff is good for the attacker. """
            if len(x.split("/")) == 3:
                wm_acc, test_acc, source_acc = x.split("/")
            elif len(x.split("/")) == 4:
                wm_acc, test_acc, runtime, source_acc = x.split("/")
            else:
                return 0

            wm_acc, test_acc, source_acc = float(wm_acc), float(test_acc), float(source_acc)
            if (wm_acc < decision_threshold) and abs(test_acc - source_acc) < max_stealing_loss:
                # Successful attack! High payoff for high test acc.
                return test_acc * 10000 - wm_acc

            # Prefer high test accuracy over low watermark retention
            return test_acc - wm_acc  # Select highest test accuracy

        return df_copy.applymap(compute_payoff)

    def get_equilibrium(self, attacks: List[str], defenses: List[str], return_best_defense=False):
        submatrix = self.slice(attacks, defenses).dropna()
        payoff = self.compute_payoff_matrix(submatrix)

        # Compute minimum over all columns and maximum over all rows.
        data_arr = payoff.to_numpy()

        # Ignore payoff values that come from data entries (zero, zero), as this is interpreted as NaN
        zero_idx = np.where(data_arr == 0.0)

        data_arr[zero_idx] = np.inf
        best_defense = np.argmin(np.max(data_arr, axis=0))

        data_arr[zero_idx] = -np.inf
        best_attack = np.argmax(np.min(data_arr, axis=1))

        metrics = submatrix.to_numpy()[best_attack, best_defense].split("/")

        # Get the name of the best attack.
        attack_name = ""
        for index, row in self.attack_data.iterrows():
            if index != best_attack+4:
                continue
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    attack_name = attack

        wm_acc, test_acc = metrics[:2]
        return {
            "eq_wm_acc": float(wm_acc), "eq_test_acc": float(test_acc),
            "best_attack": attack_name,
            "best_defense": payoff.columns.values[best_defense].split(".")[0].strip(),
            "best_defense_name": payoff.columns.values[best_defense],
            "best_defense_full": self.attack_data.loc[:, ["attacks", payoff.columns[best_defense]]] if return_best_defense else None
        }

    def get_embedding_times(self):
        """ Returns a dict with defenses and their embedding times in second.
        """
        data = {}
        for index, row in self.embed_times.iterrows():
            times = [float(x) for x in row[1].split(";")]
            data.setdefault(row[0], np.mean(times))
        return data

    def get_attack_times(self):
        """ Returns a dict with attacks and their runtimes in second.
        """
        data = {}
        for index, row in self.attack_times.iterrows():
            data.setdefault(row[0], float(row[1]))
        return data


class ImagenetNewParser(DataParser):
    def __init__(self, base_value=0.5, rescale=True):
        """ Reads all files associated with a dataset into a pandas frame.
        """
        print("Loading the Imagenet parser!")
        super().__init__()
        this_config = config["imagenet"]
        self.attack_data = pd.read_csv(this_config["attack_data"])
        self.embed_times = pd.read_csv(this_config["embed_times"])
        self.attack_times = pd.read_csv(this_config["attack_times"])
        self.base_value = base_value

        if rescale:
            self.attack_data = self.__rescale_wm_accs_in_frame()
            try:
                filename = "../data/experiment_results/_imagenet_detailed_scaled.csv"
                self.attack_data.to_csv(filename)
                print(f"Made a copy of the scaled csv file to '{filename}'")
            except:
                pass

    def get_attack_list(self):
        """ Returns a list of all attacks featured in the attack_data table
        """
        return [x.strip() for x in self.attack_data.attacks.dropna().unique()]

    def normalize_wm_acc(self, x, decision_threshold):
        """ Normalizes the watermark accuracy given to a base given its decision threshold
        """
        return np.clip((1 - self.base_value) / (1 - decision_threshold) * x + (self.base_value - decision_threshold) / (
                1 - decision_threshold), 0, np.inf)

    def __rescale_wm_accs_in_frame(self):
        """ Applies the rescaling for each watermarking scheme column-wise
        """
        defense_data = self.get_defense_list(duplicates=True)

        for column in defense_data:
            decision_threshold = np.ceil(100*float(self.attack_data[column][1]))/100

            for i, value in enumerate(self.attack_data[column][4:].copy()):
                try:
                    wm_acc, test_acc, time = value.split("/")
                except:
                    print(f"Could not split value {value} in column {column}")
                    print(self.attack_data[column][4:])
                    exit()
                scaled_wm_acc = self.normalize_wm_acc(float(wm_acc), decision_threshold)
                self.attack_data[column][4 + i] = f"{scaled_wm_acc:.4f}/{test_acc}/{time}"

        return self.attack_data

    def slice(self, attacks: List[str], defenses: List[str]):
        """ Returns the equilibrium attack and params for the attack and defense lists
        """
        # Read the column names to perform the projection.
        column_names = []
        for defense in defenses:
            for column in self.get_defense_list(duplicates=True):
                if self.equals(defense, column):
                    column_names.append(column)

        # Get the indices of the rows that we are interested in.
        row_names = []
        for index, row in self.attack_data.iterrows():
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    row_names.append(index)

        return self.attack_data[column_names].iloc[row_names]

    def get_defense_list(self, duplicates=False):
        defenses = list(self.attack_data.columns[2:].unique().values)
        # Only take those that are not "double" indexed.
        if duplicates:
            res = defenses
        else:
            # Remove duplicates
            res = []
            for defense in defenses:
                if "." not in defense:
                    res.append(defense.strip())
        return res

    def get_defense_data(self, defense_str):
        """ Returns a list of data containing the watermark accuracy and test accuracy for the
        marked and unmarked model. Returns one element for each defense param config.
        """
        if not defense_str in self.get_defense_list():
            print(f"[ERROR] Could not find {defense_str} in data table.")
            return []

        defense_data = []
        # All columns. We need to iterate to obtain the secondary column indices.
        defenses = self.get_defense_list(duplicates=True)
        for defense in defenses:
            if self.equals(defense_str, defense):
                defense_data.append(
                    {
                        "decision_threshold": self.attack_data[defense][1],
                        "unmarked_wm_acc": self.attack_data[defense][2].split("/")[0],
                        "unmarked_test_acc": self.attack_data[defense][2].split("/")[1],
                        "marked_wm_acc": self.attack_data[defense][3].split("/")[0],
                        "marked_test_acc": self.attack_data[defense][3].split("/")[1]
                    }
                )
        return defense_data

    def compute_payoff_matrix(self, df: DataFrame):
        """ Computes the payoff matrix for computing the equilibrium.
        We set the payoff to zero for attacks whose stealing loss is greater than 5 percentage points.

        High payoff => attacker wins
        """
        # Cut-off for a successful attack w.r.t. the watermark accuracy
        decision_threshold = self.base_value
        # Cut-off for a successful attack w.r.t. the test accuracy (difference to source model acc)
        max_stealing_loss = 0.05

        # Match defenses of given matrix with those in attack_data.
        # Markup the given matrix with the source model's test accuracy
        all_defenses = list(df.columns.unique().values)
        df_copy = df.copy()
        for defense in all_defenses:
            marked_test_acc = self.attack_data[defense][2].split("/")[1]
            if type(df_copy[defense]) is Series:
                df_copy[defense] = df_copy[defense].map(lambda x: str(x) + "/" + marked_test_acc)
            else:
                df_copy[defense] = df_copy[defense].applymap(lambda x: str(x) + "/" + marked_test_acc)

        # Compute the payoff matrix
        def compute_payoff(x):
            """ Select lowest watermark retention, but if watermark retention is lower than decision threshold
            we select the highest accuracy. High payoff is good for the attacker. """
            if len(x.split("/")) == 3:
                wm_acc, test_acc, source_acc = x.split("/")
            elif len(x.split("/")) == 4:
                wm_acc, test_acc, runtime, source_acc = x.split("/")
            else:
                return 0

            wm_acc, test_acc, source_acc = float(wm_acc), float(test_acc), float(source_acc)
            if (wm_acc < decision_threshold) and abs(test_acc - source_acc) < max_stealing_loss:
                # Successful attack! High payoff for high test acc.
                return test_acc * 10000 - wm_acc

            # Prefer high test accuracy over low watermark retention
            return test_acc - wm_acc  # Select highest test accuracy

        return df_copy.applymap(compute_payoff)

    def get_equilibrium(self, attacks: List[str], defenses: List[str], return_best_defense=False):
        submatrix = self.slice(attacks, defenses).dropna()
        payoff = self.compute_payoff_matrix(submatrix)

        # Compute minimum over all columns and maximum over all rows.
        data_arr = payoff.to_numpy()

        # Ignore payoff values that come from data entries (zero, zero), as this is interpreted as NaN
        zero_idx = np.where(data_arr == 0.0)

        data_arr[zero_idx] = np.inf
        best_defense = np.argmin(np.max(data_arr, axis=0))

        data_arr[zero_idx] = -np.inf
        best_attack = np.argmax(np.min(data_arr, axis=1))

        metrics = submatrix.to_numpy()[best_attack, best_defense].split("/")

        # Get the name of the best attack.
        attack_name = ""
        for index, row in self.attack_data.iterrows():
            if index != best_attack+4:
                continue
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    attack_name = attack

        wm_acc, test_acc = metrics[:2]
        return {
            "eq_wm_acc": float(wm_acc), "eq_test_acc": float(test_acc),
            "best_attack": attack_name,
            "best_defense": payoff.columns.values[best_defense].split(".")[0].strip(),
            "best_defense_name": payoff.columns.values[best_defense],
            "best_defense_full": self.attack_data.loc[:, ["attacks", payoff.columns[best_defense]]] if return_best_defense else None
        }

    def get_embedding_times(self):
        """ Returns a dict with defenses and their embedding times in second.
        """
        data = {}
        for index, row in self.embed_times.iterrows():
            times = [float(x) for x in row[1].split(";")]
            data.setdefault(row[0], np.mean(times))
        return data

    def get_attack_times(self):
        """ Returns a dict with attacks and their runtimes in second.
        """
        data = {}
        for index, row in self.attack_times.iterrows():
            data.setdefault(row[0], float(row[1]))
        return data


class ImageNetParser(DataParser):
    def __init__(self, base_value=0.5, rescale=True):
        """ Reads all files associated with a dataset into a pandas frame.
        """
        print("Loading the ImageNet parser!")
        super().__init__()
        this_config = config["imagenet"]
        self.attack_data = pd.read_csv(this_config["attack_data"])
        self.embed_times = pd.read_csv(this_config["embed_times"])
        self.attack_times = pd.read_csv(this_config["attack_times"])

        self.base_value = base_value

        if rescale:
            self.attack_data = self.__rescale_wm_accs_in_frame()
            try:
                filename = "../data/experiment_results/_imagenet_detailed_scaled.csv"
                self.attack_data.to_csv(filename)
                print(f"Made a copy of the scaled csv file to '{filename}'")
            except:
                pass

    def normalize_wm_acc(self, x, decision_threshold):
        """ Normalizes the watermark accuracy given to a base given its decision threshold
        """
        return np.clip((1 - self.base_value) / (1 - decision_threshold) * x + (self.base_value - decision_threshold) / (
                1 - decision_threshold), 0, np.inf)

    def __rescale_wm_accs_in_frame(self):
        """ Applies the rescaling for each watermarking scheme column-wise
        """
        defense_data = self.get_defense_list(duplicates=True)
        print(defense_data)
        for column in defense_data:
            decision_threshold = np.ceil(100 * float(self.attack_data[column][1])) / 100

            for i, value in enumerate(self.attack_data[column][4:].copy()):
                wm_acc, test_acc = value.split("/")
                scaled_wm_acc = self.normalize_wm_acc(float(wm_acc), decision_threshold)
                self.attack_data[column][4 + i] = f"{scaled_wm_acc:.4f}/{test_acc}"
        return self.attack_data

    def get_embedding_times(self):
        """ Returns a dict with defenses and their embedding times in second.
        """
        data = {}
        for index, row in self.embed_times.iterrows():
            times = [float(x) for x in row[1].split(";")]
            data.setdefault(row[0], np.mean(times))
        return data

    def get_attack_times(self):
        """ Returns a dict with attacks and their runtimes in second.
        """
        data = {}
        for index, row in self.attack_times.iterrows():
            data.setdefault(row[0], float(row[1]))
        return data

    def get_defense_data(self, defense_str):
        """ Returns a list of data containing the watermark accuracy and test accuracy for the
        marked and unmarked model. Returns one element for each defense param config.
        """
        if not defense_str in self.get_defense_list():
            print(f"[ERROR] Could not find {defense_str} in data table.")
            return []

        defense_data = []
        # All columns. We need to iterate to obtain the secondary column indices.
        defenses = self.get_defense_list(duplicates=True)
        for defense in defenses:
            if self.equals(defense_str, defense):
                defense_data.append(
                    {
                        "decision_threshold": self.attack_data[defense][1],
                        "unmarked_wm_acc": self.attack_data[defense][2].split("/")[0],
                        "unmarked_test_acc": self.attack_data[defense][2].split("/")[1],
                        "marked_wm_acc": self.attack_data[defense][3].split("/")[0],
                        "marked_test_acc": self.attack_data[defense][3].split("/")[1]
                    }
                )
        return defense_data

    def slice(self, attacks: List[str], defenses: List[str]):
        """ Returns the equilibrium attack and params for the attack and defense lists
        """
        # Read the column names to perform the projection.
        column_names = []
        for defense in defenses:
            for column in self.get_defense_list(duplicates=True):
                if self.equals(defense, column):
                    column_names.append(column)

        # Get the indices of the rows that we are interested in.
        row_names = []
        for index, row in self.attack_data.iterrows():
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    row_names.append(index)

        return self.attack_data[column_names].iloc[row_names]

    def __get_wm_acc(self, df: DataFrame):
        """ Expects a dataframe with '/' separators between the (i) watermark acc and (ii) test acc.
        Returns a dataframe of the same shape with only the watermark accuracy.
        """
        return df.applymap(lambda x: x.split("/")[0])

    def __get_test_acc(self, df: DataFrame):
        """ Expects a dataframe with '/' separators between the (i) watermark acc and (ii) test acc.
        Returns a dataframe of the same shape with only the test accuracy.
        """
        return df.applymap(lambda x: x.split("/")[1])

    def compute_payoff_matrix(self, df: DataFrame):
        """ Computes the payoff matrix for computing the equilibrium.
        We set the payoff to zero for attacks whose stealing loss is greater than 5 percentage points.

        High payoff => attacker wins
        """
        # Cut-off for a successful attack w.r.t. the watermark accuracy
        decision_threshold = self.base_value
        # Cut-off for a successful attack w.r.t. the test accuracy (difference to source model acc)
        max_stealing_loss = 0.05

        # Match defenses of given matrix with those in attack_data.
        # Markup the given matrix with the source model's test accuracy
        all_defenses = list(df.columns.unique().values)
        df_copy = df.copy()
        for defense in all_defenses:
            marked_test_acc = self.attack_data[defense][2].split("/")[1]
            if type(df_copy[defense]) is Series:
                df_copy[defense] = df_copy[defense].map(lambda x: str(x) + "/" + marked_test_acc)
            else:
                df_copy[defense] = df_copy[defense].applymap(lambda x: str(x) + "/" + marked_test_acc)

        # Compute the payoff matrix
        def compute_payoff(x):
            """ Select lowest watermark retention, but if watermark retention is lower than decision threshold
            we select the highest accuracy. High payoff is good for the attacker. """
            if len(x.split("/")) != 3:
                return 0
            wm_acc, test_acc, source_acc = x.split("/")
            wm_acc, test_acc, source_acc = float(wm_acc), float(test_acc), float(source_acc)

            if (wm_acc < decision_threshold) and source_acc - test_acc < max_stealing_loss:
                # Successful attack! High payoff for high test acc.
                return test_acc * 10000 - wm_acc
            # Case: Attack's test accuracy is not high enough
            return test_acc - wm_acc

        payoff = df_copy.applymap(compute_payoff)
        return payoff

    def get_equilibrium(self, attacks: List[str], defenses: List[str], return_best_defense=False):
        submatrix = self.slice(attacks, defenses).dropna()
        payoff = self.compute_payoff_matrix(submatrix)

        # Compute minimum over all columns and maximum over all rows.
        data_arr = payoff.to_numpy()

        # Ignore payoff values that come from data entries (zero, zero), as this is interpreted as NaN
        zero_idx = np.where(data_arr == 0.0)

        data_arr[zero_idx] = np.inf
        best_defense = np.argmin(np.max(data_arr, axis=0))

        data_arr[zero_idx] = -np.inf
        best_attack = np.argmax(np.min(data_arr, axis=1))

        metrics = submatrix.to_numpy()[best_attack, best_defense].split("/")

        # Get the name of the best attack.
        attack_name = ""
        for index, row in self.attack_data.iterrows():
            if index != best_attack + 4:
                continue
            for attack in attacks:
                if self.equals(attack, str(row[0])):
                    attack_name = attack

        wm_acc, test_acc = metrics[:2]
        return {
            "eq_wm_acc": float(wm_acc), "eq_test_acc": float(test_acc),
            "best_attack": attack_name,
            "best_defense": payoff.columns.values[best_defense].split(".")[0].strip(),
            "best_defense_name": payoff.columns.values[best_defense],
            "best_defense_full": self.attack_data.loc[:,
                                 ["attacks", payoff.columns[best_defense]]] if return_best_defense else None
        }

    def get_attack_list(self):
        """ Returns a list of all attacks featured in the attack_data table
        """
        return [x.strip() for x in self.attack_data.attacks.dropna().unique()]

    def get_defense_list(self, duplicates=False):
        defenses = list(self.attack_data.columns[2:].unique().values)
        # Only take those that are not "double" indexed.
        if duplicates:
            res = defenses
        else:
            # Remove duplicates
            res = []
            for defense in defenses:
                if "." not in defense:
                    res.append(defense.strip())
        return res


def main():
    """ Reads the data.
    """
    ImageNetParser()


if __name__ == "__main__":
    main()
