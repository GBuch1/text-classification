import unittest

import nltk

from classifier.our_classifier.our_classifier_models import *
from classifier.our_classifier.our_classifier_models import OurFeature


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

        self.assertIn(our_set1, self.feature_set1)
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
        self.feature1 = OurFeature("contains pain", True)
        self.feature2 = OurFeature("contains terrific", True)
        self.feature3 = OurFeature("contains awful", True)

        self.feature4 = OurFeature("contains hurt", True)
        self.feature5 = OurFeature("contains amazing", True)
        self.feature6 = OurFeature("contains horrible", True)

        self.feature_set_positive1 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
                                                   known_clas="positive")
        self.feature_set_positive2 = OurFeatureSet(features={self.feature1, self.feature2, self.feature3},
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

    def test_present_features(self):
        expected1 = str("contains outstanding = True pos:neg:neutral = 9 : 2 : 0", "contains evil = True "
                                                                                   "pos:neg:neutral = 1.2:3.1:4.1",
                        "contains outstanding = True pos:neg:neutral = 1.2:3.1:4.1")
        expected2 = str("contains outstanding = True pos:neg:neutral = 9 : 2 : 0")

        self.assertEqual(expected1, OurClassifier.present_features(3))
        self.assertEqual(expected2, OurClassifier.present_features(1))
        self.assertEqual(OurClassifier.present_features(0), "Int n cannot be 0")

    def test_train_gamma(self):
        pass

    def test_train_probStore(self):
        pass

    def test_train_notNone(self):
        pass


if __name__ == '__main__':
    unittest.main()
