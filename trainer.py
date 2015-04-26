# trains a classifier based on parsed training data provided
# makes calls to Naive Bayes (NB) and Random Forest (RF) implementations based on input


# public interface for trainer; returns a classifier object based on training data
def train(algorithm, parsed_training_data):
    classifier = nb(parsed_training_data)

    return classifier