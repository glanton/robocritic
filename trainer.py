# trains a classifier based on parsed training data provided
# makes calls to Naive Bayes (NB) and Random Forest (RF) implementations based on input

# internal import
import nb

# public interface for trainer; returns a classifier object based on training data
def train(parsed_training_data, algorithm):
    print("Training classifier...")
    classifier = nb.train(parsed_training_data)

    return classifier