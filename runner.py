# runs a classifier on data passed in and returns classified data

# internal imports
import nb
import rf


# public interface for runner; returns a list of lists (classified data)
def classify(parsed_test_data, classifier):
    print("Classifying data...")

    # classify data using the algorithm specified by the provided classifier
    if classifier.get_algorithm() == "nb":
        results = nb.classify(parsed_test_data, classifier)
    elif classifier.get_algorithm() == "rf":
        results = rf.classify(parsed_test_data, classifier)
    else:
        print("Internal error: algorithm not found")
        results = []

    return results