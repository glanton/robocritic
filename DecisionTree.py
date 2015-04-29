# the Tree and Leaf classes, which are subclasses of the generic decision tree Node


# a generic node--can be either a branching Tree or a terminal Leaf
class Node:

    # requires the type of the node ("tree" or "leaf")
    def __init__(self, node_type):
        self.node_type = node_type

    # get the node's type ("tree" or "leaf")
    def get_node_type(self):
        return self.node_type


# a branching node with the feature used for branching
class Tree(Node):

    # requires the splitting feature, left branch (has feature), and right branch (does not have feature)
    def __init__(self, feature_and_index, left, right):
        Node.__init__(self, "tree")
        self.feature_and_index = feature_and_index
        self.left = left
        self.right = right

    # get the Tree's feature and index tuple
    def get_feature_and_index(self):
        return self.feature_and_index

    # get the Tree's left branch
    def get_left(self):
        return self.left

    # get the Tree's right branch
    def get_right(self):
        return self.right


# a terminal node with class "vote" data
class Leaf(Node):

    # requires a tuple of the first class and second class votes
    def __init__(self, fc_and_sc_votes):
        Node.__init__(self, "leaf")
        self.fc_and_sc_votes = fc_and_sc_votes

    # get the Leaf's first and second class votes
    def get_votes(self):
        return self.fc_and_sc_votes