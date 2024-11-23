import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

# decrapted
def plot_confusion_matrices_original(clf, X_train, y_train, X_dev, y_dev):
   """
   Plot confusion matrices for both training and dev sets
   """
   # Get predictions
   y_train_pred = clf.predict(X_train)
   y_dev_pred = clf.predict(X_dev)
   
   # Calculate confusion matrices
   cm_train = confusion_matrix(y_train, y_train_pred)
   cm_dev = confusion_matrix(y_dev, y_dev_pred)
   
   # Create figure with two subplots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Plot training confusion matrix
   sns.heatmap(cm_train, annot=True, fmt='d', ax=ax1, cmap='Blues')
   ax1.set_title('Training Set Confusion Matrix')
   ax1.set_xlabel('Predicted Label')
   ax1.set_ylabel('True Label')
   
   # Plot dev/test confusion matrix
   sns.heatmap(cm_dev, annot=True, fmt='d', ax=ax2, cmap='Blues')
   ax2.set_title('Dev Set Confusion Matrix')
   ax2.set_xlabel('Predicted Label')
   ax2.set_ylabel('True Label')
   
   plt.tight_layout()
   plt.show()
   
   # Print classification metrics for both sets
   print("Training Set Metrics:")
   print(classification_report(y_train, y_train_pred))
   print("\nDev Set Metrics:")
   print(classification_report(y_dev, y_dev_pred))


# works well for binary
def _plot_confusion_matrices(clf, X_train, y_train, X_dev, y_dev):
    """
    Plot confusion matrices for both training and dev sets with rates in the cells.
    """
    # Get predictions
    y_train_pred = clf.predict(X_train)
    y_dev_pred = clf.predict(X_dev)
    
    # Calculate confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_dev = confusion_matrix(y_dev, y_dev_pred)
    
    # Calculate rates from confusion matrices
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
    tpr_train = tp_train / (tp_train + fn_train)  # True Positive Rate
    tnr_train = tn_train / (tn_train + fp_train)  # True Negative Rate
    fpr_train = fp_train / (tn_train + fp_train)  # False Positive Rate
    fnr_train = fn_train / (tp_train + fn_train)  # False Negative Rate
    
    tn_dev, fp_dev, fn_dev, tp_dev = cm_dev.ravel()
    tpr_dev = tp_dev / (tp_dev + fn_dev)
    tnr_dev = tn_dev / (tn_dev + fp_dev)
    fpr_dev = fp_dev / (tn_dev + fp_dev)
    fnr_dev = fn_dev / (tp_dev + fn_dev)

    # Annotations for each cell
    annot_train = np.array([[f"TNR: {tnr_train:.2f}", f"FPR: {fpr_train:.2f}"],
                            [f"FNR: {fnr_train:.2f}", f"TPR: {tpr_train:.2f}"]])
    annot_dev = np.array([[f"TNR: {tnr_dev:.2f}", f"FPR: {fpr_dev:.2f}"],
                          [f"FNR: {fnr_dev:.2f}", f"TPR: {tpr_dev:.2f}"]])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training confusion matrix
    sns.heatmap(cm_train, annot=annot_train, fmt='', ax=ax1, cmap='Blues')
    ax1.set_title('Training Set Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot dev/test confusion matrix
    sns.heatmap(cm_dev, annot=annot_dev, fmt='', ax=ax2, cmap='Blues')
    ax2.set_title('Dev Set Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Print classification metrics for both sets
    print("Training Set Metrics:")
    print(classification_report(y_train, y_train_pred))
    print("\nDev Set Metrics:")
    print(classification_report(y_dev, y_dev_pred))





def plot_confusion_matrices_rate(clf, X_train, y_train, X_dev, y_dev):
    """
    Plot confusion matrices for both training and dev sets with detailed metrics.
    """
    # Get predictions
    y_train_pred = clf.predict(X_train)
    y_dev_pred = clf.predict(X_dev)
    
    # Calculate confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_dev = confusion_matrix(y_dev, y_dev_pred)
    
    # Normalize the confusion matrices for better comparison
    cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    cm_dev_normalized = cm_dev.astype('float') / cm_dev.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot normalized training confusion matrix
    sns.heatmap(cm_train_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax1)
    ax1.set_title('Training Set Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot normalized dev/test confusion matrix
    sns.heatmap(cm_dev_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax2)
    ax2.set_title('Dev Set Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Print classification metrics for both sets
    print("Training Set Metrics:")
    print(classification_report(y_train, y_train_pred, target_names=[str(lbl) for lbl in unique_labels(y_train, y_train_pred)]))
    print("\nDev Set Metrics:")
    print(classification_report(y_dev, y_dev_pred, target_names=[str(lbl) for lbl in unique_labels(y_dev, y_dev_pred)]))
