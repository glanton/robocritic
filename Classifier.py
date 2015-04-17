# the classifier class; built by running an classification algorithm on the
# training data set


class Classifier:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.influential_features = []
        self.classifier_details = []