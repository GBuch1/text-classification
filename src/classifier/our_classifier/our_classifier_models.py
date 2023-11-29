""""Implementation of a simple classifier."""

__authors__ = "Garrett Buchanan" "Darian Choi"
__credits__ = ["Mike Ryu, Garrett Buchanan, Darian Choi"]
__email__ = "gbuchanan@westmont.edu" "dchoi@westmont.edu"

from collections import defaultdict, Counter
from typing import Iterable, Any

from nltk.corpus import twitter_samples

from classifier.classifier_models import FeatureSet, AbstractClassifier, Feature


class OurFeature(Feature):
    """"TODO: MAYBE IMPLEMENT?"""
    pass


class OurFeatureSet(FeatureSet):
    """TODO: implement so that method takes in a string data type which then tokenizes the text and builds a feature
        set based on the features in that document"""

    @classmethod
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

        return cls(features, known_clas)


class OurClassifier(AbstractClassifier):
    """# TODO: Implement math portion for gamma method to classify objects based on trained data."""

    def __init__(self, class_word_counts, class_total_words, classes):
        self.classes = classes
        self.class_total_words = class_total_words
        self.class_word_counts = class_word_counts\


    def gamma(self, feat_set: OurFeatureSet) -> str:
        """"TODO: IMPLEMENT ME:"""
        """ takes in a feature set then iterates through each feature and then 
        calculates the probability of it mapping to each class"""
        ## iterate through the feature set get the probability for each word to map to a class
        ## multiply together each feature probability together for one object for one class
        ## find all probablities for each class and then take the max
        # return the max probabilities class
        likely_class = None  # this is the variable that holds the class that is most probable
        best_score = float  # float variable that determines the class based on probablity score

        for cls in self.classes:
            score = float
            for feature in feat_set.feat:  # makes the featureset readable by calling feat property on it then iterates
                word = feature.name  # extracts the word of the feature
                word_count = self.class_word_counts[cls][word]
                class_total = self.class_total_words[cls]
                if class_total != 0:
                    score *= word_count / class_total  # multiplying all the words together
            # lets say theres the word awesome 50 times in the class positive and
            # 100 positive words in the class score would be .5
            if score > best_score:  # then compare the score we got with the max score and determine
                best_score = score
                likely_class = cls

        return likely_class

    def present_features(self, top_n: int = 1) -> None:
        """TODO: IMPLEMENT ME:"""
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        #   need to have a variable with the saved probabilities for each feature in order to then list
        #   in descending order. Need variables for the feature and the score, create a subclass for this
        #   or use a dictionary with key value pairs where key = feature and value = prob score.
        pass

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """TODO: IMPLEMENT ME"""
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        
        """
        class_word_counts = defaultdict(Counter)
        class_total_words = Counter()
        classes = set()

        for feat_set in training_set:
            cls_name = feat_set.clas
            classes.add(cls_name)
            for feature in feat_set.feat:
                class_word_counts[cls_name][feature.word] += 1
                class_total_words[cls_name] += 1

        return cls(class_word_counts,class_total_words,classes)
