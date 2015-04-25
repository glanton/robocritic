# data set parser; main purpose is to convert simple data to fully-featured data ready for training or classification
# expects 2 types of input data:
#   1) classified training set data consisting of a sentence and a class at each point
#   2) unclassified test data consisting of a sentence
# returns data, now with features

# external imports
import operator
import nltk


# the maximum number of words that can be used as features (if reached included words will have greater occurrence)
_max_features = 1000


# build the complete dictionary of words found in the data and count the occurrences of each word
def _build_word_dict(validated_data):
    word_dict = {}

    data_no_headers = validated_data.pop(0)
    for row in data_no_headers:
        words = nltk.word_tokenize(row[0])
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    return word_dict


# build the list of tuples (word, count) to be used as record features, keeping within _max_features and prioritizing
# by occurrence (greatest occurrence to lowest)
def _build_feature_list(word_dict):
    feature_list = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

    return feature_list[0:_max_features]


# build the list of lists containing both feature headers and individuals records (with their associated features)
def _build_featured_data(validated_data, feature_list):
    featured_data = []

    # build header row
    featured_data.append(validated_data[0])
    featured_data[0].extend(feature_list)

    return featured_data


# manage the order of functions required to parse input data; assumes that data has already passed validation
def _manage_parse(validated_data):

    # word_dict is a dictionary of words as keys and their occurrence as values
    word_dict = _build_word_dict(validated_data)

    # feature_list is a list of (word, count) tuples, sorted by count (occurrence) from greatest to least
    feature_list = _build_feature_list(word_dict)

    # featured_data is a list of lists, with the first row being headers, and the rest being individual records
    featured_data = _build_featured_data(validated_data, feature_list)

    return featured_data


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