# the classifier class; built by running an classification algorithm on the
# training data set


class Classifier:


    # requires algorithm string and class_names tuple of strings to initialize (expects exactly 2 class names)
    def __init__(self, algorithm, class_names, classifier_features):
        self.algorithm = algorithm
        self.class_names = class_names
        self.classifier_features = classifier_features
        self.classifier_details = []


    # add the given classifier detail
    def add_classifier_detail(self, detail):
        self.classifier_details.append(detail)

    # returns the list of features used by this classifier
    def get_classifier_features(self):
        return self.classifier_features


    # def get_classifier_details(self):