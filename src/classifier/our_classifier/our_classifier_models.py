""""Implementation of a simple classifier."""

__authors__ = "Garrett Buchanan" "Darian Choi"
__credits__ = ["Mike Ryu, Garrett Buchanan, Darian Choi"]
__email__ = "gbuchanan@westmont.edu" "dchoi@westmont.edu"

from typing import Iterable

from nltk import word_tokenize

from classifier.classifier_models import FeatureSet, AbstractClassifier, Feature


class OurFeature(Feature):
    """"TODO: MAYBE IMPLEMENT?"""
    pass


class OurFeatureSet(FeatureSet):
    """TODO: implement so that method takes in a string data type which then tokenizes the text and creates features and
    adds them into a set"""

    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """TODO: IMPLEMENT ME"""
        """
        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        """make sure the input is a string and will take in a source object and a class associated 
        with object and build a set based on the values and classification"""

        # def is_string() -> source_object:
        #     """TODO: processes the text taken in from corpus and makes sure it is a string text"""
        #     if isinstance(source_object, str):
        #         return source_object
        #     else:
        #         raise ValueError("Expected source_object of type str (text data)")
        #
        # if source_object.is_string:
        #     tokens = word_tokenize(source_object)
        #     features = {Feature(token.lower(), True) for token in tokens}
        #     return cls


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
