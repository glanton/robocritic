# the classifier class; built by running an classification algorithm on the
# training data set


class Classifier:


    # requires algorithm string and class_names tuple of strings to initialize (expects exactly 2 class names)
    def __init__(self, algorithm, class_names):
        self.algorithm = algorithm
        self.class_names = class_names
        self.classifier_details = []
        self.influential_features = []


    # add the given classifier detail
    def add_classifier_detail(self, detail):
        self.classifier_details.append(detail)


    def get_classifier_details(self):