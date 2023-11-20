import unittest


class OurFeatureTest(unittest.TestCase):
    def setUp(self):
        pass
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


class OurFeatureSetTest(unittest.TestCase):
    def setUp(self):
        pass
    def test(self):
        self.assertEqual(True, False)  # add assertion here


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
