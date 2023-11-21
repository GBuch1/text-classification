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

    def test_build(self):
        our_set1 = set["contains war": True, "contains fantastic": True]
        our_set2 = set["contains terrible": False, "contains great": False]
        our_set3 = set["contains war": True, "contains terrible": False]
        our_set4 = set["contains fantastic": True, "contains great": False]

        self.assertIn(our_set1, self.feature_set1)
        self.assertIn(our_set2, self.feature_set1)
        self.assertIn(our_set3, self.feature_set3)
        self.assertIn(our_set4, self.feature_set4)


class OurAbstractClassifierTest(unittest.TestCase):
    def setUp(self):
        self.feature5 = OurFeature("contains pain", True)
        self.feature6 = OurFeature("contains terrific", True)
        self.feature7 = OurFeature("contains awful", True)
        self.feature8 = OurFeature("contains nice", True)
        self.feature9 = OurFeature("contains ball", True)
        self.feature10 = OurFeature("contains cry", False)
        self.feature11 = OurFeature("contains sad", True)
        self.feature12 = OurFeature("contains hurt", False)
        self.feature13 = OurFeature("contains evil", True)
        self.feature14 = OurFeature("contains bottle", False)
        self.feature15 = OurFeature("contains expressionless", False)
        self.feature16 = OurFeature("contains constant", True)
        self.feature17 = OurFeature("contains mediocre", True)
        self.feature18 = OurFeature("contains wonderful", False)
        self.feature19 = OurFeature("contains ecstatic", False)
        self.feature20 = OurFeature("contains phenomenal", True)
        self.feature21 = OurFeature("contains amazing", True)
        self.feature22 = OurFeature("contains decent", True)
        self.feature23 = OurFeature("contains alright", False)
        self.feature24 = OurFeature("contains normal", True)

        self.feature_set_positive = OurFeatureSet(features={self.feature20, self.feature12})
        self.feature_set_positive2 = OurFeatureSet(features={self.feature6, self.feature8})
        self.feature_set_positive3 = OurFeatureSet(features={self.feature10, self.feature21})
        self.feature_set_positive4 = OurFeatureSet(features={self.feature7, self.feature18})
        self.feature_set_neutral = OurFeatureSet(features={self.feature9, self.feature10})
        self.feature_set_negative = OurFeatureSet(features={self.feature5, self.feature7})


    def test_gamma(self):
        featuresets = self.feature_set_positive2, self.feature_set_neutral, self.feature_set_negative
        train_set, test_set = featuresets[6:], featuresets[:6]
        classifier = OurAbstractClassifier.train(training_set=train_set)
        expected1 = "positive"
        expected2 = "neutral"
        expected3 = "negative"
        self.assertEqual(expected1, self.gamma.OurAbstractClassifier(featuresets))
        self.assertEqual(expected2, self.gamma.OurAbstractClassifier(featuresets))
        self.assertEqual(expected3, self.gamma.OurAbstractClassifier(featuresets))

    def test_present_features(self):
        pass

    def test_train(self):
        pass


if __name__ == '__main__':
    unittest.main()
