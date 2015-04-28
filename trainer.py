# trains a classifier based on parsed training data provided
# makes calls to Naive Bayes (NB) and Random Forest (RF) implementations based on input

# internal imports
import nb
import rf


# public interface for trainer; returns a classifier object based on training data
def train(parsed_training_data, algorithm):
    print("Training classifier...")

    # run naive Bayes or random forest training algorithm; else case should never fire
    if algorithm == "nb":
        classifier = nb.train(parsed_training_data)
    elif algorithm == "rf":
        classifier = rf.train(parsed_training_data)
    else:
        print("Internal error: algorithm not found")
        classifier = None

    return classifier