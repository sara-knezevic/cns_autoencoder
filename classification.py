import itertools
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def train_and_evaluate_classifier(data, labels, random_state):
    """
    Train and evaluate a classifier on the given data.
    """
    samples = data.shape[0]
    features = data.shape[1]

    # Check that data has the right shape
    if len(data.shape) != 2:
        raise ValueError("Data should be a 2D array with shape (samples, features)")

    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_roc_auc = 0
    all_confusion_matrices = []

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    fpr_list = []
    tpr_list = []

    # Perform stratified k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Train decision tree classifier with regularization
        clf = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,                # Limit the maximum depth of the tree
            min_samples_split=5,       # Minimum number of samples required to split an internal node
            min_samples_leaf=3,         # Minimum number of samples required to be at a leaf node
            max_features='sqrt'         # Number of features to consider when looking for the best split
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate classifier on test set
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        precision_scores.append(precision)
        
        recall = recall_score(y_test, y_pred)
        recall_scores.append(recall)
        
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
        
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_auc_scores.append(roc_auc)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(cm)

    # Calculate mean metrics across folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    
    # Calculate standard deviation of metrics across folds
    std_accuracy = np.std(accuracy_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_f1 = np.std(f1_scores)
    std_roc_auc = np.std(roc_auc_scores)

    # Create list for all metrics.
    metrics = [mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc]
    std_metrics = [std_accuracy, std_precision, std_recall, std_f1, std_roc_auc]

    return all_confusion_matrices, metrics, std_metrics, clf, fpr_list, tpr_list
