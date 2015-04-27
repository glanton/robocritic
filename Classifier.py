# the classifier class; built by running an classification algorithm on the training data set


class Classifier:

    # requires algorithm string and class_names tuple of strings to initialize (expects exactly 2 class names)
    def __init__(self, algorithm, class_names_counts, classifier_features):
        self.algorithm = algorithm
        self.class_names_counts = class_names_counts
        self.classifier_features = classifier_features
        self.classifier_details = []

    # add the given classifier detail
    def add_classifier_detail(self, detail):
        self.classifier_details.append(detail)

    # returns the tuple of (class name, count) tuples used by this classifier
    def get_class_names_counts(self):
        return self.class_names_counts

    # returns the list of features used by this classifier
    def get_classifier_features(self):
        return self.classifier_features

    # returns the list of details (nature of details depends on algorithm) used by this classifier
    def get_classifier_details(self):
        return self.classifier_details