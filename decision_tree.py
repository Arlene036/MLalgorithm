import argparse
import numpy as np

# 1: lest, 0: right
def read_tsv(file_path):
    with open (file_path, 'r') as f:
        lines = f.readlines()
        features_atr = lines[0].split('\t')
        lines = lines[1:]
        data = []
        for l in lines:
            data.append([int(x) for x in l.split('\t')])
        data = np.array(data)
    return data, features_atr

def output_txt(file_path, data):
    with open(file_path, 'w') as f:
        for d in data:
            f.write(str(d) + '\n')

def majority_vote(data):
    count_1 = 0
    total = 0
    for d in data:
        total += 1
        if d == 1:
            count_1 += 1
    
    if count_1 >= (total - count_1):
        return 1
    else:
        return 0

def entropy(data):
    # input an array (x,)
    if len(data) == 0:
        return -1
    count_1 = 0
    total = 0
    for d in data:
        total += 1
        if d == 1:
            count_1 += 1
    P1 = count_1 / total
    P0 = 1 - P1
    if P1 == 0 or P0 == 0:
        return 0
    else:
        return - (P1 * np.log2(P1) + P0 * np.log2(P0))

def mutual_information(data, attr_index):
    # H(A; Y) = H(Y) - P(A=0)H(Y|A=0) - P(A=1)H(Y|A=1)
    if len(data) == 0:
        return -1
    HY = entropy(data[:, -1])
    count_1 = 0
    total = 0
    for d in data[:, attr_index]:
        total += 1
        if d == 1:
            count_1 += 1
    P1 = count_1 / total
    P0 = 1 - P1
    HYA0 = entropy(data[data[:, attr_index] == 0][:,-1])
    HYA1 = entropy(data[data[:, attr_index] == 1][:,-1])
    return HY - P0 * HYA0 - P1 * HYA1

def best_attribute(data):
    # go through all feature columns
    max_mi = -1
    best_attr = None
    num_features = data.shape[1] - 1
    for i in range(num_features):
        mi = mutual_information(data, i)
        if mi > max_mi:
            max_mi = mi
            best_attr = i
    if max_mi <= 0:
        return None
    return best_attr


def all_feature_same(data):
    data_nolabel = data[:, :-1] # get all features columns
    judge_array = (data_nolabel == data_nolabel[0])
    return judge_array.all()

def all_label_same(data):
    label = data[:, -1]
    return len(set(label)) == 1

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None # column index
        self.vote = None

def train(data, max_depth):
    """
    data: numpy array. n * d. so shape[0] == n, shape[1] == d
    """
    return tree_recurse(data, 0, max_depth)

def tree_recurse(data, depth, max_depth):
    node = Node()
    # base case: no data / all labels in data are the same / all features in data are identical / stopping criterion (max depth)
    if len(data) == 0 or all_label_same(data) or all_feature_same(data) or depth >= max_depth:
        node.vote = majority_vote(data[:, -1]) # majority vote
    else: # recursion to build tree!
        # first find best attribute to split on
        split_attr_index = best_attribute(data)
        if split_attr_index is None: 
            node.vote = majority_vote(data[:, -1])
        else:
            node.attr = split_attr_index
            for i in range(2): # assume 0, 1 only in attribute
                data_i = data[data[:, split_attr_index] == i]
                if len(data_i) == 0:
                    node.vote = majority_vote(data[:, -1]) # both node.vote and node.attr exist <<< ok?
                else:
                    if i == 1:
                        node.left = tree_recurse(data_i, depth + 1, max_depth)
                    else:
                        node.right = tree_recurse(data_i, depth + 1, max_depth)
    return node

def predict(node, example):
    # example is an array, shape (d,)
    if node.vote is not None:
        return node.vote   
    else:
        print('>>>')
        print(node.attr)
        print(example[node.attr])
        print('>>>')
        if example[node.attr] == 1:
            return predict(node.left, example)
        else:
            return predict(node.right, example)

def calculate_error(groudtruth, pred):
    assert len(groudtruth) == len(pred)
    correct = 0
    total = 0
    l = len(groudtruth)
    for i in range(l):
        total += 1
        if groudtruth[i] == pred[i]:
            correct += 1
    return (total - correct) / total

def print_tree(root, data, feature_names, file):
    write_lines = []

    def dfs_print(node, data, previous_attr, previous_attr_val, depth):
        if node is None:
            return
        count_0 = np.sum(data[:, -1] == 0)
        count_1 = np.sum(data[:, -1] == 1)
        attr_info = ''
        if previous_attr is not None:
            attr_info = str(feature_names[previous_attr]) + ' = ' + str(previous_attr_val) + ': '
        write = '| '*depth + attr_info + '[' + str(count_0) + ' 0/ ' + str(count_1) +' 1]'
        write_lines.append(write)
        if node.vote is None: # not leaf
            print('leaf')
            dfs_print(node.right, data[data[:, node.attr]==0], node.attr, 0, depth + 1)
            dfs_print(node.left, data[data[:, node.attr]==1], node.attr, 1, depth + 1)

    dfs_print(root, data, None, None, 0)
    with open(file, 'w') as f:
        for line in write_lines:
            f.write(line + '\n')

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, default = 'education_train.tsv', help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, default = 'education_test.tsv', help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, default = 0,
                        help='maximum depth to which the tree should be built')
    # parser.add_argument("train_out", type=str, default='out/train_out',
    #                     help='path to output .txt file to which the feature extractions on the training data should be written')
    # parser.add_argument("test_out", type=str, default='out/test_out',
    #                     help='path to output .txt file to which the feature extractions on the test data should be written')
    # parser.add_argument("metrics_out", type=str, default='out/metrics_out',
    #                     help='path of the output .txt file to which metrics such as train and test error should be written')
    # parser.add_argument("print_out", type=str, default='out/print_out.txt',
    #                     help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    #Here's an example of how to use argparse
    # print_out = args.print_out

    args.train_out = 'out/train_out'
    args.test_out = 'out/test_out'
    args.metrics_out = 'out/metrics_out'
    args.train_out =  args.train_out  + '_' + args.train_input.split('_')[0]  + str(args.max_depth) + '.txt'
    args.test_out = args.test_out + '_'  + args.test_input.split('_')[0]  + str(args.max_depth) + '.txt'
    args.metrics_out = args.metrics_out + '_' + args.test_input.split('_')[0] + str(args.max_depth) + '.txt'

    # read data
    train_data, features_atr = read_tsv(args.train_input)
    test_data, _ = read_tsv(args.test_input)
    
    # train
    dTree = train(train_data, args.max_depth)

    # predict
    train_predict = []
    for i in range(len(train_data)):
        pred = predict(dTree, train_data[i])
        train_predict.append(pred)
    output_txt(args.train_out, train_predict)

    test_predict = []
    for i in range(len(test_data)):
        pred = predict(dTree, test_data[i])
        test_predict.append(pred)
    output_txt(args.test_out, test_predict)

    # metrics
    error_train = calculate_error(train_data[:, -1], train_predict)
    error_test = calculate_error(test_data[:, -1], test_predict)
    with open(args.metrics_out, 'w') as f:
        f.write(f'error(train): {error_train:.6f}\n')
        f.write(f'error(test): {error_test:.6f}')

    #Here is a recommended way to print the tree to a file
    # print_tree(dTree, train_data, features_atr, print_out)
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)