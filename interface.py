# user interface

# internal imports
import parser
import runner
import trainer

# external imports
import csv
import os


# available algorithms
_algorithms = ["nb", "rf"]


# function that reads a CSV file and returns it as a list of lists (each row is an element in the outer list and each
# cell is an element in the inner list
def _read_csv(filename):
    csv_data = []

    with open(filename, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)

    return csv_data


# write a CSV file with the results of the most recent test data classification (can also write other sequenced data)
def _write_csv(filename, results):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


# validates command terms for the train command, returning True if validation passes and False if it fails
def _validate_train_commands(command_list):

    # check that correct number of commands (3) were given
    if len(command_list) == 3:

        # check that an available algorithm was selected
        if command_list[1] in _algorithms:

            # check that specified file exists
            if os.path.exists("input_data/" + command_list[2]):
                return True
            else:
                print("Training data file not found")
        else:
            print("Unavailable algorithm specified; should be nb or rf")
    else:
        print("Incorrect number of command terms; should be exactly three (e.g. train rf my_data.csv)")
        print("Filename must not contain spaces")

    # if any command validation did not pass, return False
    return False


# function that orders the training process, calling parser before the trainer
def _train(command_list):

    # validate command list
    if _validate_train_commands(command_list):
        algorithm = command_list[1]
        filename = "input_data/" + command_list[2]

        # read training data from csv
        training_data = _read_csv(filename)

        # parse training data to build features and build classifier, catching errors returned by parser
        parsed_training_data = parser.prepare_training_data(training_data)
        if type(parsed_training_data) is str:
            print(parsed_training_data)
            return None
        else:
            classifier = trainer.train(parsed_training_data, algorithm)

            print("Training complete")

            # debugging output of classifier details
            # currently only works for naive Bayes... need tree-to-string to print random forest
            # _write_csv("output_data/classifier_details.csv", classifier.classifier_details)

            return classifier
    else:
        return None


# validates command terms for the train command, returning True if validation passes and False if it fails
def _validate_classify_commands(command_list):

    # check that correct number of commands (2) were given
    if len(command_list) == 2:

        # check that specified file exists
        if os.path.exists("input_data/" + command_list[1]):
            return True
        else:
            print("Test data file not found")
    else:
        print("Incorrect number of command terms; should be exactly two (e.g. classify my_data.csv)")
        print("Filename must not contain spaces")

    # if any command validation did not pass, return False
    return False


# function that orders the classification process, calling parser before the classifier
def _classify(command_list, classifier):

    # validate command list
    if _validate_classify_commands(command_list):
        filename = "input_data/" + command_list[1]

        # check that classifier is loaded
        if classifier:

            # read test data from csv
            test_data = _read_csv(filename)

            # parse test data to build features and classify; catching errors returned by parser
            parsed_test_data = parser.prepare_test_data(test_data, classifier)
            if type(parsed_test_data) is str:
                print(parsed_test_data)
            else:
                results = runner.classify(parsed_test_data, classifier)

                # output classified test data to csv
                print("Classification complete and output to classified_test_data.csv")
                _write_csv("output_data/classified_test_data.csv", results)
        else:
            print("No classifier loaded")

    return None


# interface command loop; controls what interface-level functions the user can call
def _command_loop(current_classifier):
    command = input("command: ")
    command_list = command.split(" ")

    # match on first term of command list and pass the terms forward
    if command_list[0] == "train":
        print("Training selected...")
        current_classifier = _train(command_list)
    elif command_list[0] == "classify":
        print("Classification selected...")
        _classify(command_list, current_classifier)
    elif command_list[0] == "quit":
        print("Robocritic quitting...")
        raise SystemExit
    else:
        print("Unrecognized command")

    print("--------")
    _command_loop(current_classifier)


# starts up with welcome/help text and begins the command loop
def startup():
    print("\n")
    print("Robocritic initialized.")
    print("Command line active:")
    print("train (nb | rf) file_name.csv  ||  classify file_name.csv  ||  quit")
    print("-------------------------------------------------")

    _command_loop(None)