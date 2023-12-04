""""Implementation of a simple classifier."""

__authors__ = "Garrett Buchanan" "Darian Choi"
__credits__ = ["Mike Ryu, Garrett Buchanan, Darian Choi"]
__email__ = "gbuchanan@westmont.edu" "dchoi@westmont.edu"

# SNAIL
import re
from collections import defaultdict, Counter
from itertools import islice
from math import log10
from typing import Iterable, Any, Tuple

from nltk.corpus import twitter_samples

from classifier.classifier_models import FeatureSet, AbstractClassifier, Feature


class OurFeature(Feature):
    pass


class OurFeatureSet(FeatureSet):

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """
        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        features = set()

        if isinstance(source_object, str):

            words = source_object.lower().split()
            for word in words:
                token = len(word)
                features.add(Feature(name=f"length of word {token}", value=True))

            for word in words:
                if word.isalpha() and len(word) > 1 or word == ":)" or word == ":(":
                    features.add(
                        Feature(name=f"contains word {word}", value=True))

        return cls(features, known_clas)


class OurClassifier(AbstractClassifier):

    def __init__(self, class_word_counts, class_total_words, classes, feature_probabilities):
        self.classes = classes
        self.class_total_words = class_total_words
        self.class_word_counts = class_word_counts
        self.feature_probabilities = feature_probabilities
        self.stored_ratio = {}

    def gamma(self, feat_set: OurFeatureSet) -> str:
        """ takes in a feature set then iterates through each feature and then
        calculates the probability of it mapping to each class"""
        # gamma needs to take in a specific feature set CALL the probablities for the specific class for each feature from feature probabilities variable
        # take the highest probablity of the three classes
        # using log addition we add the probablities together

        highest_probability = 0.0
        predicted_class = ''
        for cls in self.classes:
            probability = 0.0  # Probability for the current class
            for feature in feat_set.feat:
                if feature.name not in self.feature_probabilities:
                    pass
                else:
                    feature_prob = self.feature_probabilities[feature.name][cls]  # access the probability
                    probability += log10(feature_prob + 1)

                    if probability > highest_probability:
                        highest_probability = probability
                        predicted_class = cls
        return predicted_class

    ## make a ratio method
    ## takes a positive and negative probablity for a feature and then the ratio between them
    ## the greater the ratio the greater informativeness it is
    ## just to find a way to sort all the ratio probablities with the keys with it

    # def ratio(self):

    # for feature in self.feature_probabilities.keys():
    #     positive_prob = self.feature_probabilities[feature]["positive"]
    #     negative_prob = self.feature_probabilities[feature]["negative"]
    #     if positive_prob > negative_prob:
    #         ratio = positive_prob / negative_prob
    #         self.stored_ratio.update({feature: ratio})
    #     else:
    #         ratio = negative_prob / positive_prob
    #         self.stored_ratio.update({feature: ratio})

    def return_present_features(self, top_n: int = 1) -> str:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        #   need to have a variable with the saved probabilities for each feature in order to then list
        #   in descending order. Need variables for the feature and the score, create a subclass for this
        #   or use a dictionary with key value pairs where key = feature and value = prob score.
        ratio_list = []

        for feature, class_probabilities in self.feature_probabilities.items():
            positive_prob = class_probabilities["positive"]
            negative_prob = class_probabilities["negative"]
            if positive_prob == 0 or negative_prob == 0:
                continue

            if negative_prob >= positive_prob:
                ratio = negative_prob / positive_prob
                formatted_ratio = ("neg:pos", ratio)
            else:
                ratio = positive_prob / negative_prob
                formatted_ratio = ("pos:neg", ratio)

            ratio_list.append((feature, formatted_ratio))

        sorted_ratios = sorted(ratio_list, key=lambda item: item[1][1], reverse=True)

        top_n_ratios = sorted_ratios[:top_n]
        return_values = ""
        for feature, ratio in top_n_ratios:
            ratio_label, ratio_value = ratio
            return_value = f"{feature}: {ratio_label} = {ratio_value:.2f}:1\n"
            return_values += return_value
        return return_values

    def present_features(self, top_n: int = 1) -> None:
        print(self.return_present_features(top_n))

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed

        """
        class_word_counts = defaultdict(Counter)
        class_total_words = {"positive": 0, "negative": 0}
        classes = ("positive", "negative")
        feature_probabilities = {}
        if training_set is None:
            raise ValueError
        else:
            for feat_set in training_set:
                cls_name = feat_set.clas
                for feature in feat_set.feat:
                    class_word_counts[cls_name][feature.name] += 1
                    class_total_words[cls_name] += 1

            for unique in classes:
                for feat_set in training_set:  # Loop through all feature sets again
                    for feature in feat_set.feat:
                        word = feature.name  # Extract the word of the feature
                        word_count = class_word_counts[unique][word]
                        class_total = class_total_words[unique]
                        if class_total != 0:
                            score = word_count / class_total
                            if feature.name not in feature_probabilities:
                                feature_probabilities[feature.name] = {}
                            feature_probabilities[feature.name][unique] = score
                        else:
                            score = 0
                            if feature.name not in feature_probabilities:
                                feature_probabilities[feature.name] = {}
                            feature_probabilities[feature.name][unique] = score

                # come up with a way to store a dictionary with feature name and class as key
                # value should store the probablity
        # print(feature_probabilities)
        return cls(class_word_counts, class_total_words, classes, feature_probabilities)
