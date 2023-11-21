import unittest
from classifier.our_classifier.our_classifier_models import *
from classifier.our_classifier.our_classifier_models import OurFeature


class OurFeatureTest(unittest.TestCase):
    def setUp(self):
        self.new_feature1 = OurFeature("Word", True)
        self.new_feature2 = OurFeature("word2", False)
        self.new_feature3 = OurFeature("Happy", 10)
        self.new_feature4 = OurFeature("Gilmore", 20)

    def test_setup(self):
        name1 = "Word"
        value2 = False
        self.assertEqual(name1, self.new_feature1.name)
        self.assertEqual(value2, self.new_feature2.value)

    def test_compute_feature(self):
        tup = ("Word", True)
        self.assertEqual(tup, self.new_feature1.compute_feature())


class OurFeatureSetTest(unittest.TestCase):
    def setUp(self):
        self.new_feature1 = OurFeature("Word", True)
        self.new_feature2 = OurFeature("word2", False)
        self.new_feature3 = OurFeature("Happy", 19)
        self.new_feature4 = OurFeature("Gilmore", 96)
        self.new_feature_set1 = OurFeatureSet(features={self.new_feature1, self.new_feature2})
        self.new_feature_set2 = OurFeatureSet(features={self.new_feature3, self.new_feature4})

    def test_has_features(self):
        features1 = ("Word", True)
        self.assertIn(features1, self.new_feature_set1)


class OurAbstractClassfierTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_gamma(self):
        pass

    def test_present_features(self):
        pass

    def test_train(self):
        pass


if __name__ == '__main__':
    unittest.main()
