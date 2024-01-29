import numpy as np
import os
os.system("cls")


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, result=None):
        #for decision node 
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        #for leaf node
        self.result = result

class Decisiontree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    
    def split_data(self, x, feature, value):
        left_indices = np.where(x[:,feature] <= value)[0]
        right_indices = np.where(x[:,feature] > value)[0]
        return left_indices, right_indices
    
    def entropy(self, y):
        p = np.bincount(y) / len(y)
        return -np.sum(p*np.log2(p))
    
    def info_gain(self, y, left_indices, right_indices):
        parent_impurity = self.entropy(y)
        left_impurity = self.entropy(y[left_indices])
        right_impurity = self.entropy(y[right_indices])
        w_left = len(left_indices) / len(y)
        w_right = len(right_indices) / len(y)
        information_gain = parent_impurity - w_left*left_impurity - w_right*right_impurity
        return information_gain
    
    def fit(self, x, y, depth=0):

        if depth == self.max_depth or np.all(y[0]==y):
            return Node(result=y[0])
        
        num_sampels, num_feature = x.shape
        best_info_gain = 0.0
        best_split = None
        for i in range(num_feature):
            feature_values = x[:,i]
            unique_values = np.unique(feature_values)
            for j in range(unique_values):
                left_indices, right_indices = self.split_data(x, i, j)
                information_gain = self.info_gain(y, left_indices, right_indices)
                if information_gain > best_info_gain:
                    best_info_gain = information_gain
                    best_split = (i, j, left_indices, right_indices)
        if best_info_gain == 0.0:
            return Node(result=np.bincount(y).argmax())
        
        feature, value, left_indices, right_indices = best_split
        left_subtree = self.fit(x[left_indices], y[left_indices], depth+1)
        right_subtree = self.fit(x[right_indices], y[right_indices], depth+1)
        self.tree = Node(feature=feature, value=value, left=left_subtree, right=right_subtree)
        return self.tree
    

    def predict_recursive(self, x, node):
        if node.result is not None:
            return node.result
        
        if x[node.feature] <= node.value:
            return self.predict_recursive(x, node.left)
        else:
            return self.predict_recursive(x, node.right)
        
    def predict(self, x):
        result = [self.predict_recursive(i, self.tree) for i in x]
        return np.array(result)