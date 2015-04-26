# implementation of the Naive Bayes algorithm

# internal import
import Classifier


# public interface function to start training process; expects 2D list with
# binary features as input
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
    classifier = Classifier.Classifier("nb", (first_class, second_class), classifier_features)
    for detail in classifier_details:
        classifier.add_classifier_detail(detail)

    return classifier
