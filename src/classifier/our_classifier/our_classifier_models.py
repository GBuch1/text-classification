"Data types for classifier inheriting abstract classes from classifier models"


from classifier.classifier_models import Feature, FeatureSet, AbstractClassifier

__author__ = "Darian Choi, Garrett Buchanan"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "dchoi@westmont.edu, gbuchanan@westmont.edu"




class OurFeature(Feature):
    pass

class OurFeatureSet(FeatureSet):
    pass

class OurAbstractClassifier(AbstractClassifier):
    pass