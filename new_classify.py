import numpy as np
import pandas as pd
from tqdm import  tqdm

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 用于分割的特征索引
        self.threshold = threshold          # 分割阈值
        self.left = left                    # 左子树
        self.right = right                  # 右子树
        self.value = value                  # 叶节点的类别值

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth                # 最大深度
        self.min_samples_split = min_samples_split  # 节点分裂的最小样本数
        self.root = None                          # 树的根节点

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # 终止条件
        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            value = self._most_common_label(y)
            return Node(value=value)

        # 寻找最佳分割特征和阈值
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        # 分裂数据集并构建左右子树
        left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # 返回当前节点
        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _gini_impurity(self, left_labels, right_labels):
        num_left = len(left_labels)
        num_right = len(right_labels)
        total_samples = num_left + num_right

        p_left = np.sum(left_labels != self._most_common_label(left_labels)) / num_left if num_left > 0 else 0
        p_right = np.sum(right_labels != self._most_common_label(right_labels)) / num_right if num_right > 0 else 0

        gini = (num_left / total_samples) * p_left + (num_right / total_samples) * p_right
        return gini

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict_tree(x, self.root) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
# 加载数据

class RandomForest:
    def __init__(self, n_estimators=40, max_depth=10, min_samples_split=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in tqdm(range(self.n_estimators)):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]) * x_i)
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # 计算x与所有训练样本的距离
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # 获取距离最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取这k个样本的类别
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回最常见的类别作为预测结果
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        self.parameters = []

        # 计算每个类别的均值和标准差
        for c in self.classes:
            X_c = X[y == c]
            self.parameters.append({
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0) + 1e-8  # 加上一个小的常数以避免除以零
            })

    def _calculate_likelihood(self, mean, std, x):
        # 计算高斯概率密度函数
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return np.prod((1 / (np.sqrt(2 * np.pi) * std)) * exponent)

    def _calculate_prior(self, c):
        # 计算先验概率
        return np.mean(self.y_train == c)

    def _classify(self, sample):
        # 对于每个类别，计算后验概率，并返回概率最大的类别
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = self._calculate_prior(c)
            likelihood = self._calculate_likelihood(self.parameters[i]['mean'], self.parameters[i]['std'], sample)
            posterior = prior * likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        # 对所有样本进行分类
        y_pred = [self._classify(sample) for sample in X]
        return np.array(y_pred)


# 创建SVM模型并进行训练

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[0]
    # print(X.shape)
    # print(num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_samples = int(test_size * num_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    # print(X.iloc[:10])
    # print(test_indices,train_indices)

    X_train, X_test =  X.iloc[test_samples:], X.iloc[:test_samples]
    y_train, y_test = y.iloc[test_samples:], y.iloc[:test_samples]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # print(type(X_train),X_train[:10])

    return X_train, X_test, y_train, y_test


train_data = pd.read_csv("data_feature.csv")

# 选择特征和目标变量
X = train_data[['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']]
y = train_data['ground_truth_key']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X_train,y_train)
# 创建随机森林分类器

# svm = SVM()
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# 在训练集上训练模型


# clf = RandomForest()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# knn = KNN()
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)

bys = NaiveBayes()
bys.fit(X_train,y_train)
y_pred = bys.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
