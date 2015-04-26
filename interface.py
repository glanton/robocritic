# user interface

# internal imports
import parser
import runner
import trainer

# external imports
import csv


# function that reads a CSV file and returns it as a list of lists (each row is an element in the outer list and each
# cell is an element in the inner list
def _read_csv(filename):
    csv_data = []

    with open(filename, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)

    return csv_data


# function that orders the training process, calling parser before the trainer
def _train():
    training_data = _read_csv("input_data/training_data.csv")
    parsed_training_data = parser.prepare_training_data(training_data)
    classifier = trainer.train("nb", parsed_training_data)

    return classifier


# function that orders the classification process, calling parser before the classifier
def _classify(classifier):
    test_data = _read_csv("input_data/test_data.csv")
    parsed_test_data = parser.prepare_test_data(test_data, classifier)
    results = runner.classify(classifier, parsed_test_data)

    return results


# interface command loop; controls what interface-level functions the user can call
def _command_loop(current_classifier):
    command = input("command: ")

    if command == "train":
        print("Training initiated...")
        current_classifier = _train()
    elif command == "classify":
        print("Classification commenced...")
        results = _classify(current_classifier)
    elif command == "quit":
        print("Robocritic quitting...")
        raise SystemExit
    else:
        print("Unrecognized command")

    print("--------")
    _command_loop(current_classifier)


# starts up with welcome/help text and begins the command loop
def startup():
    print("\n")
    print("I am Robocritic.")
    print("You may command me thus.")
    print("train | classify | save | load | quit")
    print("--------------------------------------------")

    _command_loop(None)