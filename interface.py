# user interface

import parser
import nb
import runner


# function that orders the training process, calling parser before the trainer
def _train():
    training_data = "placeholder training data"
    parsed_training_data = parser.prepare_data(training_data)
    classifier = nb.train(parsed_training_data)

    return classifier


# function that orders the classification process, calling parser before the classifier
def _classify(classifier):
    test_data = "placeholder test data"
    parsed_test_data = parser.prepare_data(test_data)
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