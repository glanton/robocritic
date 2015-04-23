# data set parser; main purpose is to convert simple data to fully-featured data ready for training or classification
# expects 2 types of input data:
#   1) classified training set data consisting of a sentence and a class at each point
#   2) unclassified test data consisting of a sentence
# returns data, now with features

# external imports
import nltk


# manage the order of functions required to parse input data; assumes that data has already passed validation
def _manage_parse(validated_data):
    # will contain a list of all words founds in the input data
    word_list = []

    data_no_headers = validated_data.pop(0)
    for row in data_no_headers:


# one of two public interfaces for parser; returns an updated list of lists of parsed training data; if incoming data
# is not formatted correctly returns a descriptive error in a string for the interface to handle
def prepare_training_data(unparsed_data):
    if not unparsed_data:
        return "input data is empty"
    elif len(unparsed_data[0]) < 2:
        return "input data columns missing; should be exactly 2"
    elif len(unparsed_data[0]) > 2:
        return "input data too many columns; should be exactly 2"
    elif (unparsed_data[0][0] != "RECORD") or (unparsed_data[0][1] != "CLASS"):
        return "input data headers incorrect; should be 'RECORD', 'CLASS'"
    else:
        return _manage_parse(unparsed_data)


# one of two public interfaces for parser; returns an updated list of lists of parsed test data; if incoming data
# is not formatted correctly returns a descriptive error in a string for the interface to handle
def prepare_test_data(unparsed_data):
    if not unparsed_data:
        return "input data is empty"
    elif len(unparsed_data[0]) > 1:
        return "input data too many columns; should be exactly 1"
    elif unparsed_data[0][0] != "RECORD":
        return "input data header incorrect; should be 'RECORD'"
    else:
        return _manage_parse(unparsed_data)