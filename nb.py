# implementation of the Naive Bayes algorithm
# to keep things more readable, "fc" is used to indicate "first class", and "sc" to indicate "second class"

# internal import
import Classifier
import debug


# public interface function to training a classifier; expects 2D list with binary features as input
def train(parsed_training_data):
    first_class = parsed_training_data[1][1]
    second_class = ""

    # build structure of details with feature names and build list of classifier features
    classifier_features = []
    classifier_details = []
    for feature in parsed_training_data[0][2:]:
        detail_list = [feature, 0, 0]
        classifier_details.append(detail_list)
        classifier_features.append(feature)

    # capture the two class names and count their totals
    class_counts = {}
    for row in parsed_training_data[1:]:

        # debug counter
        debug.run_counter("nb.train", 10)

        # count the occurrences of each class and add to the class counts dictionary
        class_name = row[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            second_class = class_name
            class_counts[class_name] = 1

        # increment the respective class count within the detail tuple
        for i in range(0, len(classifier_details)):
            if row[i+2]:
                if class_name == first_class:
                    classifier_details[i][1] += 1
                else:
                    classifier_details[i][2] += 1

    # calculate the probability of each feature occurring based on its class occurring
    for detail_tuple in classifier_details:
        detail_tuple[1] = detail_tuple[1] / class_counts[first_class]
        detail_tuple[2] = detail_tuple[2] / class_counts[second_class]

    # create new instance of Classifier and populate with classifier details
    class_names_counts = ((first_class, class_counts[first_class]), (second_class, class_counts[second_class]))
    classifier = Classifier.Classifier("nb", class_names_counts, classifier_features)
    for detail in classifier_details:
        classifier.add_classifier_detail(detail)

    return classifier


# public interface function to classify data; expects 2D list with binary features as input and a classifier object
def classify(parsed_test_data, classifier):

    # HELPER FUNCTION for classify function
    # -----------------------------------

    # calculate the probability of a class by multiplying along a list of individual feature probabilities
    def _calculate_probability(feature_prob_list):
        calc_prob = 1
        for prob in feature_prob_list:
            calc_prob = calc_prob * prob

        return calc_prob

    # ----------------------------------
    # MAIN CODE for classify function
    # ----------------------------------

    # get class names and counts from classifier
    fc_name = classifier.get_class_names_counts()[0][0]
    fc_count = classifier.get_class_names_counts()[0][1]
    sc_name = classifier.get_class_names_counts()[1][0]
    sc_count = classifier.get_class_names_counts()[1][1]

    # calculate the probability that a record is in the first class or the second class
    total_count = fc_count + sc_count
    fc_prob = fc_count / total_count
    sc_prob = sc_count / total_count

    # build results with assigned classes by calculating the more probable class based on a record's features
    results = []
    classifier_details = classifier.get_classifier_details()
    for row in parsed_test_data[1:]:

        # debug counter
        debug.run_counter("nb.classify", 10)

        # feature-by-feature lists of probabilities that the record belongs to the first or second classes
        # first probability in the lists is the overall probability that a record belongs to that class
        fc_feature_prob_list = [fc_prob]
        sc_feature_prob_list = [sc_prob]

        # for each feature in a record (row), add the corresponding probability from classifier details to the
        # feature-by-feature probability lists
        for i in range(0, len(classifier_details)):
            feature = row[i+1]
            if feature:
                fc_feature_prob = classifier_details[i][1]
                sc_feature_prob = classifier_details[i][2]
                fc_feature_prob_list.append(fc_feature_prob)
                sc_feature_prob_list.append(sc_feature_prob)
            else:
                fc_feature_prob = 1 - classifier_details[i][1]
                sc_feature_prob = 1 - classifier_details[i][2]
                fc_feature_prob_list.append(fc_feature_prob)
                sc_feature_prob_list.append(sc_feature_prob)

        # calculate probability that the record belongs to the first and second classes
        fc_calc_prob = _calculate_probability(fc_feature_prob_list)
        sc_calc_prob = _calculate_probability(sc_feature_prob_list)

        # build results by assigning the class with greater calculated probability to this record
        # if calculations are equal, assign the overall most probable class
        record = row[0]
        if fc_calc_prob > sc_calc_prob:
            results.append((record, fc_name))
        elif sc_calc_prob > fc_calc_prob:
            results.append((record, sc_name))
        else:
            if fc_prob > sc_prob:
                results.append((record, fc_name))
            else:
                results.append((record, sc_name))

    return results