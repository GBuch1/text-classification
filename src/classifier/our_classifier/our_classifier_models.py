""""Implementation of a simple classifier."""

__authors__ = "Garrett Buchanan" "Darian Choi"
__credits__ = ["Mike Ryu, Garrett Buchanan, Darian Choi"]
__email__ = "gbuchanan@westmont.edu" "dchoi@westmont.edu"

from typing import Iterable, Any
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from collections import Counter

from classifier.classifier_models import FeatureSet, AbstractClassifier, Feature


class OurFeature(Feature):
    """"TODO: MAYBE IMPLEMENT?"""
    pass


class OurFeatureSet(FeatureSet):
    """TODO: implement so that method takes in a string data type which then tokenizes the text and builds a feature
        set based on the features in that document"""

    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """TODO: IMPLEMENT ME"""
        """
        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        features = set()

        if isinstance(source_object, str):
            tokens = source_object.split()
            for token in tokens:
                features.add(Feature(name=f"word_{token}", value=1))  # Create a feature for each token

        return cls


class OurAbstractClassifier(AbstractClassifier):
    """# TODO: Implement math portion for gamma method to classify objects based on trained data."""

    def __init__(self):
        pass

    def gamma(self, feat_set: OurFeatureSet) -> str:
        """"TODO: IMPLEMENT ME:"""
        """ takes in a feature set then iterates through each feature and then 
        calculates the probability of it mapping to each class"""
        pass

    def present_features(self, top_n: int = 1) -> None:
        """TODO: IMPLEMENT ME:"""
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        pass

    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """TODO: IMPLEMENT ME"""
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        pass

    pass
