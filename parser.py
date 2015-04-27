# data set parser; main purpose is to convert simple data to fully-featured data ready for training or classification
# expects 2 types of input data:
#   1) classified training set data consisting of a sentence and a class at each point
#   2) unclassified test data consisting of a sentence
# returns data, now with features

# external imports
import nltk
import operator
import string


# the maximum number of words that can be used as features (if reached included words will have greater occurrence)
_max_features = 10000

# the minimum ratio of word's occurrence to the number of training records (occurrences / number of records)
_min_occ_ratio = 0.0005


# build the complete dictionary of words found in the data and count the occurrences of each word
def _build_word_dict(validated_data):
    word_dict = {}

    # tokenize list of words in each record (skipping header row); only count words once per record (no duplicates) and
    # strip out punctuation
    for row in validated_data[1:]:
        words_with_duplicates = nltk.word_tokenize(row[0])
        words = set(words_with_duplicates)
        for word in words:
            if word not in string.punctuation:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

    return word_dict


# build the list of words to be used as record features, keeping within _max_features and prioritizing
# by occurrence (greatest occurrence to lowest)
def _build_feature_list(word_dict, data_length):
    word_count_list = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

    # build feature list from sorted word count list until the occurrence ratio falls below the minimum
    feature_list = []
    for feature in word_count_list:
        occ_ratio = feature[1] / data_length
        if occ_ratio >= _min_occ_ratio:
            feature_list.append(feature[0])
        else:
            break

    return feature_list[0:_max_features]


# build the list of lists containing both feature headers and individuals records (with their associated features)
def _build_featured_data(validated_data, feature_list):
    featured_data = []

    # build header row
    header_row = list(validated_data[0])
    header_row.extend(feature_list)
    featured_data.append(header_row)

    # count for debugging purposes
    count = 0

    # build data rows, checking each feature against each record's list of words; a 1 in the matching column means the
    # feature exists; a 0 means that it does not (columns will be either RECORD, CLASS, [FEATURE_0], [FEATURE_1], etc.
    # in the case of training data or RECORD, [FEATURE_0], [FEATURE_1], etc. in the case of test data; thus if a record
    # has FEATURE_1 and is training data the value of the 4th item in that row's list would be 1
    for row in validated_data[1:]:

        # print count and increment for debugging purposes
        if count % 100 == 0:
            print("_build_featured_data: " + str(count))
        count += 1

        row_feature_list = []

        words = nltk.word_tokenize(row[0])
        for feature in feature_list:
            if feature in words:
                row_feature_list.append(1)
            else:
                row_feature_list.append(0)

        # append the build data row to the total featured data
        current_row = list(row)
        current_row.extend(row_feature_list)
        featured_data.append(current_row)

    return featured_data


# manage the order of functions required to parse input training data; assumes that data has already passed validation
def _manage_training_parse(validated_data):

    # word_dict is a dictionary of each word that occurs in the data as keys and their occurrence count as values
    word_dict = _build_word_dict(validated_data)

    # feature_list is a list of words, sorted by count (occurrence) from greatest to least
    data_length = len(validated_data) - 1
    feature_list = _build_feature_list(word_dict, data_length)

    # featured_data is a list of lists, with the first row being headers, and the rest being individual records
    featured_data = _build_featured_data(validated_data, feature_list)

    return featured_data


# manage the order of functions required to parse input test data; assumes that data has already passed validation
def _manage_test_parse(validated_data, classifier):

    # feature_list is a list of words, sorted by count (occurrence) from greatest to least
    feature_list = classifier.get_classifier_features()

    # featured_data is a list of lists, with the first row being headers, and the rest being individual records
    featured_data = _build_featured_data(validated_data, feature_list)

    return featured_data


# validates whether a class has been assigned to each record, and whether exactly 2 classes have been used
# returns True if classes are incorrect
def _classes_incorrect(unparsed_data):
    class_list = []

    # for each class in data, make sure that it is one of exactly 2 classes
    for row in unparsed_data[1:]:
        class_name = row[1]
        if class_name not in class_list:
            if len(class_list) < 2:
                class_list.append(class_name)
            else:
                return True

    # make sure that at exactly 2 classes were found
    if len(class_list) == 2:
        return False
    else:
        return True


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
    elif not unparsed_data[1]:
        return "input data missing records"
    elif _classes_incorrect(unparsed_data):
        return "input data classes are inconsistent; each record should be assigned 1 of exactly 2 classes"
    else:
        print("Parsing training data...")
        return _manage_training_parse(unparsed_data)


# one of two public interfaces for parser; returns an updated list of lists of parsed test data; if incoming data
# is not formatted correctly returns a descriptive error in a string for the interface to handle
def prepare_test_data(unparsed_data, classifier):
    if not unparsed_data:
        return "input data is empty"
    elif len(unparsed_data[0]) > 1:
        return "input data too many columns; should be exactly 1"
    elif unparsed_data[0][0] != "RECORD":
        return "input data header incorrect; should be 'RECORD'"
    else:
        print("Parsing test data...")
        return _manage_test_parse(unparsed_data, classifier)