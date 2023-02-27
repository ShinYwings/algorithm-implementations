from numpy import true_divide
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Directed, Connected graph with one root node.
  # Every other node has a single predecessor (parent)
  # and no or more successors (children)
  # leaves : Nodes without successors
  # All nodes are connected by edges
  # Depth of a node is the # of edges on the path to the root
  # height of the whole tree: the # of edges on the longest path from the root to any half

  # Root : Initial decision node
  # Node : Internal decision node for testing on an attribute
  # Edge : Rule to follow (*)
  # Leaf : Terminal node represents the resulting classification

  # CHAID
  # handles missing values by treating them as a separate valid category and doesn't perform any pruning

  # Iterative Dichotomiser 3 (ID3)
  # splitting criterion
  # It doesn't apply any pruning procedure nor does it handle numeric attributes or missing values.
  # 지금 내가 가진 데이터는 continuous data이므로 이거 사용 안해도 되겠다.

  # Classification and Regression Tree (CART)
  # regression tree (내가 필요한 것)

  # Complexity
  # training instances  : n
  # attributes          : m
  # Building a tree     : O(m*nlogn)
  # Subtree replacement : O(n)
  # Subtree raising     : O(n*(nlogn)^2)
  #   - Every instance may have to be redistributed at every node between its leaf and the root : O(nlogn)
  #   - Cost for redistribution : O(log n)

class Node:
    def __init__(self, desc):
        
        self.threshold = desc[0]
        self.attribute = desc[1]
        self.total_sample = desc[2]
        self.gini = desc[3]
        self.lchild = []
        self.rchild = []
        self.isleaf = False
        self.terminal_class = None
        
class DecisionTree():
        
    def __init__(self, leafCond=0.000001, max_depth = 10):
        self.root = None
        self.leafCond = leafCond
        self.max_depth = max_depth
        
    def find(self, val):
        return self.findNode([self.root], val)
        
    def findNode(self, currentNode, val):
        # print("currentNode.terminal_class->",currentNode[0].terminal_class)
        if (currentNode[0] == None):
            return False
        elif (currentNode[0].isleaf):
            return currentNode[0].terminal_class
        elif (val[currentNode[0].attribute] <= currentNode[0].threshold):
            return self.findNode(currentNode[0].lchild, val)
        else:
            return self.findNode(currentNode[0].rchild, val)
    
    def growingTree(self, feature, label, depth=1):
        # 2.35, sepal_length, total sample(105), setosa 몇개~~ , 0.0023122, 왼쪽 섭셋 정보, 오른쪽 섭셋 정보
        # winner index=['threshold', 'attribute', 'total_sample', 'gini'])
        winner, label_desc, left_desc, right_desc = split(feature=feature, label=label)
        
        self.root = Node(winner)

        if self.leafCond > self.root.gini:
            # print('label_desc.max()->',label_desc.idxmax())
            self.root.terminal_class = label_desc.idxmax()
            self.root.isleaf = True
            
        else:
            self.insertNode(self.root.lchild, left_desc[0], left_desc[1], depth)
            self.insertNode(self.root.rchild, right_desc[0], right_desc[1], depth)
        
        # self.print(depth,self.root)
        
    def insertNode(self, currentNode, feature, label, depth):
        
        if depth == self.max_depth:
            return

        depth = depth+1
        
        winner, label_desc, left_desc, right_desc = split(feature=feature, label=label)
        
        currentNode.append(Node(winner))
        
        if self.leafCond > currentNode[0].gini:
            # print('label_desc.max()->',label_desc.idxmax())
            currentNode[0].terminal_class = label_desc.idxmax()
            currentNode[0].isleaf = True
        else:
            self.insertNode(currentNode[0].lchild, left_desc[0], left_desc[1], depth)
            self.insertNode(currentNode[0].rchild, right_desc[0], right_desc[1], depth)
        
        # self.print(depth,currentNode)

    def traverse(self):
        return self.traverseNode([self.root])

    def traverseNode(self, currentNode):
        result = []
        if (currentNode[0].lchild):
            result.extend(self.traverseNode(currentNode[0].lchild))
        if (currentNode[0] is not None):
            result.extend([currentNode[0].attribute])
        if (currentNode[0].rchild):
            result.extend(self.traverseNode(currentNode[0].rchild))
        return result
    
    def print(self, depth, node):
        print("=============level:{}===============".format(depth))
        print("node.threshold->", node.threshold)
        print("node.attribute->", node.attribute)
        print("node.total_sample->", node.total_sample)
        print("node.gini->", node.gini)
        print("node.terminal_class->", node.terminal_class)
        
def split(feature, label):
    
    # find Gains, growingTree, splitting Criterion
    columns = feature.shape[1]
    m = feature.shape[0]
    
    min_gini = 1 # initialize the optimized gini value
    for i in range(0, columns):
        column_name = feature.columns[i] # e.g. sepal_length
        # print(column_name)

        classes = feature[column_name].value_counts() # counts of each values
        # print("len(classes)->",len(classes))
        
        for j in range(0, len(classes)):
            current_class = classes.keys().tolist()[j]
            # print(column_name,"->",current_class)
            
            G_left_subset = feature[feature[column_name] <= current_class] # sepal_length가 5.0보다 같거나 작은 것들을 오른쪽 가지
            G_left_label = label[feature[column_name] <= current_class]

            G_right_subset = feature[feature[column_name] > current_class] # sepal_length가 5.0보다 큰 것들을 왼쪽 가지
            G_right_label = label[feature[column_name] > current_class]
            
            m_left = G_left_subset.shape[0]
            m_right = G_right_subset.shape[0]
            m_left_probability = m_left / m
            m_right_probability = m_right / m
            
            G_left_label_hit = G_left_label.value_counts().tolist() # setosa versicolor virginica 
            
            G_right_label_hit = G_right_label.value_counts().tolist() # setosa virginica versicolor
            
            # Gini_index is used in the classficiation problem
            # Gini impurity describes how homogeneous or "pure" a node is
            # If subgini == 0 then a node is pure
            # If subgini == 1 then a node with many samples from many different classes
            # i.e. subgini는 hit rate가 높을수록 값이 저점 낮아짐
            
            left_gini = 1
            for k in range(0, len(G_left_label_hit)):
                left_gini = left_gini - (G_left_label_hit[k]/m_left)**2
            
            right_gini = 1
            for k in range(0, len(G_right_label_hit)):
                right_gini = right_gini - (G_right_label_hit[k]/m_right)**2
                
            # gain이 0에 가까울수록 정보량이 줄어듬 (fitting됨)
            gini = m_left_probability*left_gini + m_right_probability*right_gini
            
            if gini < min_gini:
                min_gini = gini
                
                winner = [current_class, column_name, m, min_gini]
                label_desc = pd.value_counts(label)
                left_desc = [G_left_subset, G_left_label]
                right_desc = [G_right_subset, G_right_label]
                
    if min_gini == 1:
        print("cost function is not worked, exit.")
        exit(0)       
    
    return winner, label_desc, left_desc, right_desc
    # END find Gains

def decisionTree_train(feature, label):
    
    model = []
    
    tree = DecisionTree()
    tree.growingTree(feature, label)
    
    model.append(tree)
    
    return model

def decisionTree_test(model, feature):

    prediction = []
    
    tree = model[0]
    
    for i in range(len(feature.index)):
        prediction.append(tree.find(feature.iloc[i]))
    
    return prediction

def decisionTree_eval(prediction, label):
    
    accuracy = 0
    # same = [1 if i == j else 0 for i, j in zip(prediction, label.tolist())]
    # print(same)
    for i,j in zip(prediction, label.tolist()):
        if i==j:
            accuracy += 1        
    
    accuracy /= len(prediction) 
    return accuracy

def decisionTree_analysis(model):

    feature_hist = [0,0,0,0,0]
    
    tree = model[0]
    res = tree.traverse()
    feature_hist[0] = res.count('sepal_length')
    feature_hist[1] = res.count('sepal_width')
    feature_hist[2] = res.count('petal_length')
    feature_hist[3] = res.count('petal_width')
    feature_hist[4] = res.count('species')
    
    return feature_hist

### Decision Tree

if __name__=="__main__":
    # Load the iris dataset
    iris = sns.load_dataset("iris")
    # Print the counts for the class-wise samples
    pd.value_counts(iris.species)
    # Description for iris dataset
    iris.describe()
    # Training Set Configuration
    train = iris.sample(frac=0.7, random_state=500)
    # Test Set Configuration
    test = iris.drop(train.index)
    # Label remove
    X_train = train.drop(labels='species', axis=1)
    # Label stack
    y_train = train.species
    
    # Label remove
    X_test = test.drop(labels='species', axis=1)
    # Label stack
    y_test = test.species
    
    # Training
    tree_model = decisionTree_train(X_train,y_train)

    # Prediction
    results = decisionTree_test(tree_model, X_test)
    print(results[:5])

    # # Evaluation
    accuracy = decisionTree_eval(results, y_test)
    print('Test accuracy: ' + str(accuracy))

    # Analysis of Model
    feature_hist = decisionTree_analysis(tree_model)
    plt.bar(iris.axes[1][:], feature_hist)
    plt.title('Number of usages', fontsize=20)
    plt.show()