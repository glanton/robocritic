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
    def __init__(self, feature, left, right):
        Node.__init__(self, "tree")
        self.feature = feature
        self.left = left
        self.right = right


# a terminal node with class "vote" data
class Leaf(Node):

    # requires the first class and second class counts
    def __init__(self, fc_count, sc_count):
        Node.__init__(self, "leaf")
        self.fc_count = fc_count
        self.sc_count = sc_count

