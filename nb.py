# implementation of the Naive Bayes algorithm

# internal import
import Classifier


# public interface function to training a classifer; expects 2D list with binary features as input
def train(parsed_training_data):
    class_counts = {}
    first_class = parsed_training_data[1][1]
    second_class = ""
    classifier_features = []
    classifier_details = []

    # build structure of details with feature names and build list of classifier features
    for feature in parsed_training_data[0][2:]:
        detail_list = [feature, 0, 0]
        classifier_details.append(detail_list)
        classifier_features.append(feature)

    # capture the two class names and count their totals
    for row in parsed_training_data[1:]:
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
    class_name_counts = ((first_class, class_counts[first_class]), (second_class, class_counts[second_class]))
    classifier = Classifier.Classifier("nb", class_name_counts, classifier_features)
    for detail in classifier_details:
        classifier.add_classifier_detail(detail)

    return classifier

# public interface function to classify data; expects 2D list with binary features as input and a classifier object
def classify(parsed_test_data, classifier):
    results = []
    classifier_details = classifier.get_classifier_details()
    first_class_count = classifier.get_class_names_counts()[0]
    second_class_count = classifier.get_class_names_counts()[1]

    # calculate the probability that a record is in the first class or the second class
    total_count = first_class_count[1] + second_class_count[1]
    first_class_prob = first_class_count[1] / total_count
    second_class_prob = second_class_count[1] / total_count

    for row in parsed_test_data[1:]:
        # initialize the list of probabilities that the row belongs to the first or second classes by adding the overall
        # probability of a record belonging to that class
        first_class_prob_list = [first_class_prob]
        second_class_prob_list = [second_class_prob]

        for i in range(0, len(classifier_details)):
            feature = row[i+1]
            if feature:
                first_class_feature_prob = classifier_details[i][1]
                second_class_feature_prob = classifier_details[i][1]
                first_class_prob_list.append(first_class_feature_prob)
                second_class_prob_list.append(second_class_feature_prob)
            else:
                first_class_feature_prob = 1 / classifier_details[i][1]
                second_class_feature_prob = 1 / classifier_details[i][1]
                first_class_prob_list.append(first_class_feature_prob)
                second_class_prob_list.append(second_class_feature_prob)

        # calculate probability that record belongs to the first class by multiplying all probability terms
        first_class_calc_prob = 1
        for prob in first_class_prob_list:
            first_class_calc_prob = first_class_calc_prob * prob

        # calculate probability that record belongs to the second class by multiplying all probability terms
        second_class_calc_prob = 1
        for prob in second_class_prob_list:
            second_class_calc_prob = second_class_calc_prob * prob

        if first_class_calc_prob > second_class_calc_prob:
            record_class = (row[0], first_class_count[0])
            results.append(record_class)
        else:
            record_class = (row[0], second_class_count[0])
            results.append(record_class)

    return results