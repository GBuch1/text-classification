import unittest

import nltk

from classifier.our_classifier.our_classifier_models import *


class OurFeatureTest(unittest.TestCase):
    def setUp(self):
        self.feature1 = OurFeature("contains war", True)
        self.feature2 = OurFeature("contains fantastic", True)
        self.feature3 = OurFeature("contains terrible", False)
        self.feature4 = OurFeature("contains great", False)

    def test_setup(self):
        name1 = "contains war"
        name2 = "contains great"
        value1 = True
        value2 = False
        self.assertEqual(name1, self.feature1.name)
        self.assertEqual(value1, self.feature1.value)
        self.assertEqual(name2, self.feature4.name)
        self.assertEqual(value2, self.feature4.value)


class OurFeatureSetTest(unittest.TestCase):
    def setUp(self):
        self.feature1 = OurFeature("contains war", True)
        self.feature2 = OurFeature("contains fantastic", True)
        self.feature3 = OurFeature("contains terrible", False)
        self.feature4 = OurFeature("contains great", False)
        self.feature_set1 = OurFeatureSet(features={self.feature1, self.feature2})
        self.feature_set2 = OurFeatureSet(features={self.feature3, self.feature4})
        self.feature_set3 = OurFeatureSet(features={self.feature1, self.feature3})
        self.feature_set4 = OurFeatureSet(features={self.feature2, self.feature4})

        self.feature5 = OurFeature("contains hate", True)
        self.feature6 = OurFeature("contains terrific", True)
        self.feature7 = OurFeature("contains awful", False)
        self.feature8 = OurFeature("contains nice", False)
        self.feature_set5 = OurFeatureSet(features={self.feature5, self.feature6})
        self.feature_set6 = OurFeatureSet(features={self.feature7, self.feature8})
        self.feature_set7 = OurFeatureSet(features={self.feature5, self.feature7})
        self.feature_set8 = OurFeatureSet(features={self.feature6, self.feature8})

        self.feature9 = OurFeature("contains incredible", None)
        self.feature10 = OurFeature(None, True)
        self.feature_set9 = OurFeatureSet(features={self.feature9, self.feature10})

    def test_build(self):
        our_set1 = set["contains war": True, "contains fantastic": True]
        our_set2 = set["contains terrible": False, "contains great": False]
        our_set3 = set["contains war": True, "contains terrible": False]
        our_set4 = set["contains fantastic": True, "contains great": False]

        self.assertEqual(our_set1, self.feature_set1.feat)
        self.assertIn(our_set2, self.feature_set1)
        self.assertIn(our_set3, self.feature_set3)
        self.assertIn(our_set4, self.feature_set4)

    def test_build2(self):
        our_set5 = set["contains hate": True, "contains terrific": True]
        our_set6 = set["contains awful": False, "contains nice": False]
        our_set7 = set["contains hate": True, "contains awful": False]
        our_set8 = set["contains terrific": True, "contains nice": False]

        self.assertIn(our_set5, self.feature_set5)
        self.assertIn(our_set6, self.feature_set6)
        self.assertIn(our_set7, self.feature_set7)
        self.assertIn(our_set8, self.feature_set8)

    def test_none(self):
        our_set9 = set["contains incredible": None, None, True]
        self.assertEqual(our_set9, self.feature_set9)


class OurAbstractClassifierTest(unittest.TestCase):
    def setUp(self):
        self.feature5 = OurFeature("contains pain", True)
        self.feature6 = OurFeature("contains terrific", True)
        self.feature7 = OurFeature("contains awful", True)
        self.feature8 = OurFeature("contains nice", True)
        self.feature9 = OurFeature("contains ball", True)
        self.feature10 = OurFeature("contains cry", True)
        self.feature11 = OurFeature("contains sad", True)
        self.feature12 = OurFeature("contains hurt", True)
        self.feature13 = OurFeature("contains evil", True)
        self.feature14 = OurFeature("contains bottle", True)
        self.feature15 = OurFeature("contains expressionless", True)
        self.feature16 = OurFeature("contains constant", True)
        self.feature17 = OurFeature("contains mediocre", True)
        self.feature18 = OurFeature("contains wonderful", True)
        self.feature19 = OurFeature("contains ecstatic", True)
        self.feature20 = OurFeature("contains phenomenal", True)
        self.feature21 = OurFeature("contains amazing", True)
        self.feature22 = OurFeature("contains decent", True)
        self.feature23 = OurFeature("contains alright", False)
        self.feature24 = OurFeature("contains normal", True)

        self.feature_set_positive1 = OurFeatureSet(features={self.feature20, self.feature18})
        self.feature_set_positive2 = OurFeatureSet(features={self.feature6, self.feature8})
        self.feature_set_positive3 = OurFeatureSet(features={self.feature6, self.feature21})
        self.feature_set_positive4 = OurFeatureSet(features={self.feature19, self.feature18})
        self.feature_set_neutral = OurFeatureSet(features={self.feature21, self.feature13})
        self.feature_set_negative = OurFeatureSet(features={self.feature5, self.feature7})

        self.feature1 = OurFeature("contains pain", True)
        self.feature2 = OurFeature("contains terrific", True)
        self.feature3 = OurFeature("contains awful", True)

        self.feature4 = OurFeature("contains hurt", True)
        self.feature5 = OurFeature("contains amazing", True)
        self.feature6 = OurFeature("contains horrible", True)

        self.feature_set_present1 = OurFeatureSet(features={self.feature5, self.feature6, self.feature8},
                                                   known_clas="positive")
        self.feature_set_present2 = OurFeatureSet(features={self.feature11, self.feature7, self.feature6},
                                                   known_clas="positive")
        self.feature_set_neutral1 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
                                                  known_clas="neutral")
        self.feature_set_neutral2 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
                                                  known_clas="neutral")
        self.feature_set_negative1 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
                                                   known_clas="negative")
        self.feature_set_negative2 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
                                                   known_clas="negative")

        self.feature_set_positive3 = OurFeatureSet(features={self.feature4, self.feature5, self.feature6},
                                                   known_clas="positive")
        self.feature_set_neutral3 = OurFeatureSet(features={self.feature4, self.feature5, self.feature6},
                                                  known_clas="neutral")
        self.feature_set_negative3 = OurFeatureSet(features={self.feature4, self.feature5, self.feature6},
                                                   known_clas="negative")
        """---train test setup---"""

        self.feature7 = OurFeature("contains bad", True)
        self.feature8 = OurFeature("contains fantastic", True)
        self.feature9 = OurFeature("contains negative", True)

        self.feature10 = OurFeature("contains cry", True)
        self.feature11 = OurFeature("contains positive", True)
        self.feature12 = OurFeature("contains devastating", True)

        self.feature_set_positive4 = OurFeatureSet(features={self.feature6, self.feature20, self.feature24},
                                                   known_clas="positive")
        self.feature_set_positive5 = OurFeatureSet(features={self.feature7, self.feature8, self.feature9},
                                                   known_clas="positive")
        self.feature_set_neutral4 = OurFeatureSet(features={self.feature7, self.feature8, self.feature10, self.feature18},
                                                  known_clas="neutral")
        self.feature_set_neutral5 = OurFeatureSet(features={self.feature7, self.feature8, self.feature9},
                                                  known_clas="neutral")
        self.feature_set_negative4 = OurFeatureSet(features={self.feature5, self.feature7, self.feature9},
                                                   known_clas="negative")
        self.feature_set_negative5 = OurFeatureSet(features={self.feature7, self.feature8, self.feature9},
                                                   known_clas="negative")

        self.feature_set_positive6 = OurFeatureSet(features={self.feature7, self.feature8, self.feature9},
                                                   known_clas="positive")

    def test_gamma(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2]
        classifier = OurClassifier.train(featuresets)
        expected = "positive"
        self.assertEqual(expected, classifier.gamma(self.feature_set_positive4))

    def test_gamma_positive(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2, self.feature_set_neutral1,
                       self.feature_set_neutral2, self.feature_set_negative1, self.feature_set_negative2]
        classifier = OurClassifier.train(training_set=featuresets)
        expected1 = "positive"
        self.assertEqual(expected1, classifier.gamma(a_feature_set=self.feature_set_positive3))

    def test_gamma_neutral(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2, self.feature_set_neutral1,
                       self.feature_set_neutral2, self.feature_set_negative1, self.feature_set_negative2]
        classifier = OurClassifier.train(training_set=featuresets)
        expected1 = "neutral"
        self.assertEqual(expected1, classifier.gamma(a_feature_set=self.feature_set_neutral3))

    def test_gamma_negative(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2, self.feature_set_neutral1,
                       self.feature_set_neutral2, self.feature_set_negative1, self.feature_set_negative2]
        classifier = OurClassifier.train(training_set=featuresets)
        expected1 = "negative"
        self.assertEqual(expected1, classifier.gamma(a_feature_set=self.feature_set_negative3))

    def test_present_features_multiple(self):
        featuresets = [self.feature_set_present1, self.feature_set_present2]
        classifier = OurClassifier.train(featuresets)
        expected_output1 = {"contains pain": {'positive': 0.2, 'negative': 0.8},
                            "contains awful": {'positive': 0.1, 'negative': 0.9},
                            "contains terrific": {'positive': 0.75, 'negative': 0.25}}
        true_output1 = classifier.present_features(top_n=3)

        self.assertEqual(expected_output1, true_output1)

    def test_present_features_single(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2]
        classifier = OurClassifier.train(featuresets)
        expected_output2 = {'contains terrific': {'positive': 0.4, 'negative': 0.6}}
        true_output2 = classifier.present_features(top_n=1)

        self.assertEqual(expected_output2, true_output2)

    def test_present_features_error(self):
        featuresets = [self.feature_set_positive1, self.feature_set_positive2]
        classifier = OurClassifier.train(featuresets)
        true_output3 = classifier.present_features(top_n=0)
        if true_output3:
            self.assertRaises(ValueError)

    def test_train_gamma(self):
        """If gamma returns the expected results then we know that our classifier is being trained correctly"""
        featuresets = [self.feature_set_positive3, self.feature_set_positive4]
        classifier = OurClassifier.train(featuresets)
        expected = "positive"
        self.assertEqual(expected, classifier.gamma(self.feature_set_positive1))

    # def test_train_probVariable(self):
    #     """Only one test here for the method probability_value() because not sure if it will be implemented
    #        in the final code. For now using it as one of the three necessary tests for the train method.
    #        This method if implemented would store the probability value of the gamma equation before gamma
    #        returns the predicted class as a string."""
    #     featuresets = [self.feature_set_positive4, self.feature_set_positive5, self.feature_set_neutral4,
    #                    self.feature_set_neutral5, self.feature_set_negative4, self.feature_set_negative5]
    #     classifier = OurClassifier.train(training_set=featuresets)
    #     expected_prob = 0.0326758
    #     self.assertAlmostEqual(expected_prob,
    #                            classifier.gamma.probability_value(a_feature_set=self.feature_set_positive6))

    def test_train_empty(self):
        featuresets = None
        with self.assertRaises(ValueError):
            OurClassifier.train(training_set=featuresets)


if __name__ == '__main__':
    unittest.main()
