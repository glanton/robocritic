# implementation of the Naive Bayes algorithm

# internal import
import Classifier


# public interface function to start training process; expects 2D list with
# binary features as input
def train(parsed_training_data):
    class_counts = {}
    first_class = parsed_training_data[1][1]
    second_class = ""
    classifier_details = []

    # build structure of details with feature names
    for feature_tuple in parsed_training_data[0][2:]:
        detail_tuple = (feature_tuple[0], 0, 0)
        classifier_details.append(detail_tuple)

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
    for feature_tuple in classifier_details:
        feature_tuple[1] = feature_tuple[1] / class_counts[first_class]
        feature_tuple[2] = feature_tuple[2] / class_counts[second_class]

    # create new instance of Classifier and populate with classifier details
    classifier = Classifier("nb", (first_class, second_class))
    for detail in classifier_details:
        classifier.add_classifier_detail(detail)

    return classifier
