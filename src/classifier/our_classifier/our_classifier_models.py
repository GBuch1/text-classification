""""Implementation of a simple classifier."""

__authors__ = "Garrett Buchanan" "Darian Choi"
__credits__ = ["Mike Ryu, Garrett Buchanan, Darian Choi"]
__email__ = "gbuchanan@westmont.edu" "dchoi@westmont.edu"

from typing import Iterable

from classifier.classifier_models import FeatureSet, AbstractClassifier, Feature


class OurFeature(Feature):
    def compute_feature(self):
        pass


class OurFeatureSet(FeatureSet):
    """"""

    def processing_text(self):
        pass

    def processing_numerical(self):
        pass

    def __init__(self):
        pass

    def build(self):
        pass

    pass


class OurAbstractClassifier(AbstractClassifier):
    """"""

    def __init__(self):
        pass

    def gamma(self, feat_set: OurFeatureSet) -> str:
        """Input self and a feature set created by
        implementing the FeatureSet class."""
        pass

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        pass

    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        pass

    pass
