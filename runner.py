# runs a classifier on data passed in and returns classified data


# public interface for runner; returns a list of lists (classified data)
def classify(parsed_test_data, classifier):
    print("Classifying data...")
    results = nb.classify(parsed_test_data, classifier)

    return results