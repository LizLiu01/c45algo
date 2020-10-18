import math
import csv
import json

output = []


def gain(union_set, subsets, classes):
    s = len(union_set)
    information_gain_before_split = entropy(union_set, classes)
    weights = [len(subset) / s for subset in subsets]
    weighted_information_gain_after_split = 0
    for i in range(len(subsets)):
        weighted_information_gain_after_split += weights[i] * entropy(subsets[i], classes)
    total_gain = information_gain_before_split - weighted_information_gain_after_split
    return total_gain


def entropy(data_set, classes):
    n = len(data_set)
    if n == 0:
        return 0
    num_classes = [0 for i in classes]
    for row in data_set:
        class_index = list(classes).index(row[-1])
        num_classes[class_index] += 1
    num_classes = [x/n for x in num_classes]
    ent = 0
    for num in num_classes:
        ent += num * log(num)
    return ent * -1


def log(x):
    if x == 0:
        return 0
    else:
        return math.log(x, 2)


def read_csv_data(file_name):
    output = []
    with open(file_name, mode='r')as file:
        csv_file = csv.reader(file)

        for lines in csv_file:
            output.append(lines)
    return output


def read_feature_metadata(file_name):
    with open(file_name) as file:
        data = json.load(file)

    classes = data['classes']
    features = data['features']
    return classes, features


def prepare_training_data(data, features):
    num_attributes = len(features)
    for index, row in enumerate(data):
        for attr_index in range(num_attributes):
            data[index][attr_index] = float(data[index][attr_index])
    return data


def split_attribute(cur_data, cur_attributes, attributes, classes):
    split_subsets = []
    max_entropy = -1 * float("inf")
    best_feature = -1

    best_threshold = None
    for attribute in cur_attributes:
        index_of_attribute = attributes.index(attribute)

        cur_data.sort(key=lambda x: x[index_of_attribute])
        for j in range(0, len(cur_data) - 1):
            if cur_data[j][index_of_attribute] != cur_data[j + 1][index_of_attribute]:
                threshold = (cur_data[j][index_of_attribute] + cur_data[j + 1][index_of_attribute]) / 2
                less = []
                greater = []
                for row in cur_data:
                    if row[index_of_attribute] > threshold:
                        greater.append(row)
                    else:
                        less.append(row)
                e = gain(cur_data, [less, greater], classes)
                if e >= max_entropy:
                    split_subsets = [less, greater]
                    max_entropy = e
                    best_feature = attribute
                    best_threshold = threshold
    return best_feature, best_threshold, split_subsets


def generate_tree(data, remaining_features, features, classes):
    tree = generate_branches(data, remaining_features, features, classes)
    return tree


def all_same_class(data):
    for row in data:
        if row[-1] != data[0][-1]:
            return False
    return data[0][-1]


def generate_branches(data, remaining_features, features, classes):

    if all_same_class(data):
        return {}
    elif len(remaining_features) == 0:
        return {}
    else:
        (best, best_threshold, subset) = split_attribute(data, remaining_features, features, classes)

        remaining_f = remaining_features[:]
        remaining_f.remove(best)
        node = dict()
        node['feature'] = best
        node['threshold'] = best_threshold
        node['distinct_classes_left'] = list(set(map(lambda x: x[-1], subset[0])))
        node['distinct_classes_right'] = list(set(map(lambda x: x[-1], subset[1])))
        node['children'] = [generate_branches(ss, remaining_f, features, classes) for ss in subset]
        return node


def print_tree(tree):
    print_node(tree)


def is_leaf_node(node):
    return node == {}


def print_node(node, indent=""):
    if not is_leaf_node(node):

        leftChild = node['children'][0]
        rightChild = node['children'][1]

        if is_leaf_node(leftChild):
            output.append([indent, 'if', node['feature'], " <= ", str(node['threshold']), ': return', f"'{node['distinct_classes_left'][0]}'"])
        else:
            output.append([indent, 'if', node['feature'], " <= ", str(node['threshold']), ':'])
            print_node(leftChild, indent + "	")

        if is_leaf_node(rightChild):
            output.append([indent, 'if', node['feature'], " > ", str(node['threshold']), ': return', f"'{node['distinct_classes_right'][0]}'"])
        else:
            output.append([indent, 'if', node['feature'], " > ", str(node['threshold']), ':'])
            print_node(rightChild, indent + "	")

