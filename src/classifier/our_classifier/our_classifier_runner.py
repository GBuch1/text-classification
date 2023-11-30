"Runner for classifier"
import random

from nltk.corpus import twitter_samples

from our_classifier_models import OurFeature, OurFeatureSet, OurClassifier

__author__ = "Darian Choi, Garrett Buchanan"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "dchoi@westmont.edu, gbuchanan@westmont.edu"


def main():
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    #building the feature sets to use
    pos_feature_sets = [OurFeatureSet.build(tweet, 'positive') for tweet in positive_tweets]
    neg_feature_sets = [OurFeatureSet.build(tweet, 'negative') for tweet in negative_tweets]

    # Combine positive and negative feature sets
    all_featuresets = pos_feature_sets + neg_feature_sets

    # shuffle and randomize so you get authentic results
    random.shuffle(all_featuresets)
    train_set, test_set = all_featuresets[8000:], all_featuresets[:2000]

    # Train your classifier
    classifier = OurClassifier.train(train_set)

    print(classifier.present_features(30))
