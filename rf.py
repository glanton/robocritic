# implementation of the Random Forest algorithm
# to keep things more readable, "fc" is used to indicate "first class", and "sc" to indicate "second class"

# internal imports
import Classifier
import DecisionTree
import debug

# external imports
import math
import random


# the maximum number of random decision trees to train
_max_trees = 500

# the maximum number of random features to sample
_max_sample = 200

# the minimum number of records required to build another Tree node; otherwise force a Leaf node
_leaf_threshold = 5

# the maximum number of times a tree can recurse into new branches when growing (1000 is Python's default internal max)
_rec_limit = 100


# public interface function to training a classifier; expects 2D list with binary features as input
def train(parsed_training_data):

    # HELPER FUNCTIONS for train function
    # -----------------------------------

    # calculate the information entropy in a set of data with two classes
    def _calculate_entropy(fc_count, sc_count):
        if fc_count == 0 or sc_count == 0:
            entropy = 0
        else:
            fc_prob = fc_count / (fc_count + sc_count)
            sc_prob = 1 - fc_prob

            entropy = -fc_prob * math.log2(fc_prob) - sc_prob * math.log2(sc_prob)

        return entropy

    # function to take a random sample of features from provided training data and find the feature to split on
    # that provides the greatest information gain
    def _find_best_sampled_feature(training_data_cut):

        # pick a number of random samples up to square root length of classifier features, but no more than _max_sample
        features_length = len(classifier_features)
        sqrt_features = math.ceil(math.sqrt(features_length))
        sample_feature_size = sqrt_features if sqrt_features < _max_sample else _max_sample
        sample_feature_list = []
        for j in range(0, sample_feature_size):
            feature_index = random.randint(0, (features_length - 1))
            sample_feature_list.append(feature_index)

        # for each sampled feature, split the training data on that feature, count the votes of the resulting class
        # distribution, and build a list of the sampled features and their associated vote information
        vote_count_list = []
        for feature_index in sample_feature_list:
            # add 2 to feature index to skip RECORD and CLASS columns
            skip_index = feature_index + 2

            # count the resulting first class and second class votes on each side of the split on this feature
            fc_has_vote = 0
            sc_has_vote = 0
            fc_has_not_vote = 0
            sc_has_not_vote = 0
            for find_row in training_data_cut:
                if find_row[skip_index]:
                    if find_row[1] == first_class:
                        fc_has_vote += 1
                    else:
                        sc_has_vote += 1
                else:
                    if find_row[1] == first_class:
                        fc_has_not_vote += 1
                    else:
                        sc_has_not_vote += 1

            # add the vote information resulting from the split to the vote count list
            index_and_votes = (feature_index, (fc_has_vote, sc_has_vote), (fc_has_not_vote, sc_has_not_vote))
            vote_count_list.append(index_and_votes)

        # calculate the class vote entropy for the total training data cut; this can be found by adding up class votes
        # in the first (or any) index of the vote_count_list
        fc_parent_vote = vote_count_list[0][1][0] + vote_count_list[0][2][0]
        sc_parent_vote = vote_count_list[0][1][1] + vote_count_list[0][2][1]
        parent_entropy = _calculate_entropy(fc_parent_vote, sc_parent_vote)

        # initialize variable to contain information of best information gain split that can be found; use first feature
        # in vote count list as starting data; second element is information gain
        best_information_feature = (vote_count_list[0], 0)

        # calculate entropy resulting from each feature split in the vote count list to find information gain
        parent_votes = fc_parent_vote + sc_parent_vote
        for feature_vote_count in vote_count_list:
            # gather has (left) and has not (right) votes
            fc_has_vote = feature_vote_count[1][0]
            sc_has_vote = feature_vote_count[1][1]
            fc_has_not_vote = feature_vote_count[2][0]
            sc_has_not_vote = feature_vote_count[2][1]

            # calculate entropy of left (has) and right (has not) children
            child_left_entropy = _calculate_entropy(fc_has_vote, sc_has_vote)
            child_right_entropy = _calculate_entropy(fc_has_not_vote, sc_has_not_vote)

            # calculate proportion of votes in left and right children
            child_left_prop = (fc_has_vote + sc_has_vote) / parent_votes
            child_right_prop = (fc_has_not_vote + sc_has_not_vote) / parent_votes

            # calculate overall child entropy and information gain
            child_entropy = child_left_prop * child_left_entropy + child_right_prop * child_right_entropy
            information_gain = parent_entropy - child_entropy

            # compare with current best information gain
            if information_gain > best_information_feature[1]:
                best_information_feature = (feature_vote_count, information_gain)

        # prepare return data as feature and votes tuple:
        # ((feature_name, feature_index), (fc_has_vote, sc_has_vote), (fc_has_not_vote, sc_has_not_vote))
        feature_index = best_information_feature[0][0]
        feature_name = classifier_features[feature_index]
        fc_has_vote = best_information_feature[0][1][0]
        sc_has_vote = best_information_feature[0][1][1]
        fc_has_not_vote = best_information_feature[0][2][0]
        sc_has_not_vote = best_information_feature[0][2][1]
        return_data = ((feature_name, feature_index), (fc_has_vote, sc_has_vote), (fc_has_not_vote, sc_has_not_vote))

        return return_data

    # recursive function for building a random decision tree based on the provided cut of training data
    def _rec_build_random_tree(training_data_cut, rec_count):
        # increase recursion count by 1
        rec_count += 1

        # find the feature to split the data that provides greatest information gain from a random sample
        # returns tuple ((feature_name, feature_index), (fc_has_vote, sc_has_vote), (fc_has_not_vote, sc_has_not_vote))
        feature_and_votes = _find_best_sampled_feature(training_data_cut)

        # if training data falls below a preset threshold or the vote is unanimous build a Leaf node;
        # otherwise split data on feature and build a Tree node; also enforce a recursion limit
        fc_has_vote = feature_and_votes[1][0]
        sc_has_vote = feature_and_votes[1][1]
        fc_has_not_vote = feature_and_votes[2][0]
        sc_has_not_vote = feature_and_votes[2][1]

        # length of training data cut
        cut_length = len(training_data_cut)

        # build left (has feature) branch
        if cut_length < _leaf_threshold or fc_has_vote == 0 or sc_has_vote == 0 or rec_count > _rec_limit:
            # build Leaf based on votes
            left_branch = DecisionTree.Leaf((fc_has_vote, sc_has_vote))
        else:
            # split out and build Tree
            has_feature_data = []
            for tree_row in training_data_cut:
                # add 2 to feature index to skip RECORD and CLASS columns
                feature_index = feature_and_votes[0][1] + 2
                if tree_row[feature_index]:
                    has_feature_data.append(tree_row)

            # recurse into the left branch building the tree of data that has feature
            left_branch = _rec_build_random_tree(has_feature_data, rec_count)

        # build right (has not feature) branch
        if cut_length < _leaf_threshold or fc_has_not_vote == 0 or sc_has_not_vote == 0 or rec_count > _rec_limit:
            # build Leaf based on votes
            right_branch = DecisionTree.Leaf((fc_has_not_vote, sc_has_not_vote))
        else:
            # split out and build Tree
            has_not_feature_data = []
            for tree_row in training_data_cut:
                # add 2 to feature index to skip RECORD and CLASS columns
                feature_index = feature_and_votes[0][1] + 2
                if not tree_row[feature_index]:
                    has_not_feature_data.append(tree_row)

            # recurse into the right branch building the tree of data without feature
            right_branch = _rec_build_random_tree(has_not_feature_data, rec_count)

        # build tree with splitting feature name and index, and the left and right branches
        feature_name_index = feature_and_votes[0]
        random_tree = DecisionTree.Tree(feature_name_index, left_branch, right_branch)

        return random_tree

    # ----------------------------------
    # MAIN CODE for train function
    # ----------------------------------

    first_class = parsed_training_data[1][1]
    second_class = ""

    # build list of classifier's feature names
    classifier_features = []
    for feature in parsed_training_data[0][2:]:
        classifier_features.append(feature)

    # find second class; also count the occurrence of each class and add to the class counts dictionary
    class_counts = {}
    for row in parsed_training_data[1:]:
        class_name = row[1]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            second_class = class_name
            class_counts[class_name] = 1

    # build n random decision trees, where n is the length of the data, up to the max
    classifier_details = []
    training_data_length = len(parsed_training_data[1:])
    tree_count = training_data_length if training_data_length < _max_trees else _max_trees
    for i in range(0, tree_count):

        # debug counter
        debug.run_counter("rf.train", 10)

        # bootstrap training data, taking random sample of rows (with replacement) equal to the length of training data
        bootstrap_data = []
        for k in range(0, training_data_length):
            row_index = random.randint(0, (training_data_length - 1)) + 1
            bootstrap_data.append(parsed_training_data[row_index])

        # build a random decision tree by passing bootstrapped data (without headers) and list of classifier features
        # to recursive build function
        random_decision_tree = _rec_build_random_tree(bootstrap_data, 0)

        # add tree to classifier details
        classifier_details.append(random_decision_tree)

    # create new instance of Classifier and populate with classifier details
    class_names_counts = ((first_class, class_counts[first_class]), (second_class, class_counts[second_class]))
    classifier = Classifier.Classifier("rf", class_names_counts, classifier_features)
    for detail in classifier_details:
        classifier.add_classifier_detail(detail)

    # reset the debug counter
    debug.reset_counter("rf.train")

    return classifier


# public interface function to classify data; expects 2D list with binary features as input and a classifier object
def classify(parsed_test_data, classifier):

    # HELPER FUNCTION for classify function
    # -----------------------------------

    # recursively run through the given decision tree until reaching a terminal Leaf node; current_row is expected
    # without RECORD column (meaning that its feature index matches the classifier's feature list)
    def _rec_run_decision_tree(tree, current_row):

        # check node type of tree; if Tree recurse further, if Leaf return votes
        if tree.get_node_type() == "tree":
            # get the index of the Tree node's splitting feature
            feature_and_index = tree.get_feature_and_index()
            index = feature_and_index[1]

            # if the row has the feature, recurse left (has branch), otherwise recurse right (has not branch)
            if current_row[index]:
                left_branch = tree.get_left()
                return _rec_run_decision_tree(left_branch, current_row)
            else:
                right_branch = tree.get_right()
                return _rec_run_decision_tree(right_branch, current_row)
        else:
            leaf_votes = tree.get_votes()
            return leaf_votes

    # ----------------------------------
    # MAIN CODE for train function
    # ----------------------------------

    # get class names and counts from classifier
    fc_name = classifier.get_class_names_counts()[0][0]
    fc_count = classifier.get_class_names_counts()[0][1]
    sc_name = classifier.get_class_names_counts()[1][0]
    sc_count = classifier.get_class_names_counts()[1][1]

    # build results by running each row through the classifier's decision trees
    results = []
    classifier_details = classifier.get_classifier_details()
    for row in parsed_test_data[1:]:

        # debug counter
        debug.run_counter("rf.classify", 10)

        # variables to hold the total first and second class votes across all decision trees
        fc_total_votes = 0
        sc_total_votes = 0

        # run the row through each decision tree in the classifier
        for detail in classifier_details:

            # get tuple of first and second class votes from detail decision tree
            fc_and_sc_votes = _rec_run_decision_tree(detail, row[1:])

            # pull out class votes and add votes to total tallies
            fc_vote = fc_and_sc_votes[0]
            sc_vote = fc_and_sc_votes[1]
            fc_total_votes += fc_vote
            sc_total_votes += sc_vote

        # calculate more probable class (the one that received more total votes)
        # if votes are equal, assign the overall most probable class
        record = row[0]
        if fc_total_votes > sc_total_votes:
            results.append((record, fc_name))
        elif sc_total_votes > fc_total_votes:
            results.append((record, sc_name))
        else:
            if fc_count > sc_count:
                results.append((record, fc_name))
            else:
                results.append((record, sc_name))

    # reset the debug counter
    debug.reset_counter("rf.classify")

    return results