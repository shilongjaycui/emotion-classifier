import numpy as np
from numpy import ndarray
from pandas import DataFrame
from typing import List, Dict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import load

from data import DATASET, EMOTION_DICT, set_display_options
from train_model import MODEL_FNAME

# Set display options
set_display_options()

X_test: List = DATASET['test']['text']
y_test: List = DATASET['test']['label']

def examine_misclassifications(df: DataFrame, predicted_label: str, actual_label: str) -> None:
    predicted_label: int = int(predicted_label)
    actual_label: int = int(actual_label)
    print(f"Examining {EMOTION_DICT[actual_label]} ({actual_label}) sentences that were misclassified as {EMOTION_DICT[predicted_label]} ({predicted_label})...")
    condition: bool = (df["predicted_label"] == predicted_label) & (df["actual_label"] == actual_label)
    print(df[condition])

if __name__ == '__main__':
    model: Pipeline = load(filename=MODEL_FNAME)

    print(f'dataset column names: {DATASET.column_names}\n')

    y_pred: ndarray = model.predict(X=X_test)  # predict the actual class
    print(f"accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}\n")
    print(f"classification report:\n{classification_report(y_test, y_pred)}\n")

    # Get predicted probabilities for the positive class
    y_scores: ndarray = model.predict_proba(X=X_test)  # predict the class probabilities
    print(f'y_scores.shape: {y_scores.shape}\n')

    print("Creating a precision-recall curve...")
    try:
        precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_scores)
    except ValueError:
        print('`precision_recall_curve` is designed for binary classification problems.')
        print('ChatGPT: While precision-recall curves are commonly associated with binary classification problems, they can be extended to multi-class scenarios by considering each class as a separate binary classification problem.')
    
    precision_dict: Dict = {}
    recall_dict: Dict = {}
    average_precision: Dict = {}

    plt.figure(figsize=(8, 5))

    for i in range(len(model.classes_)):
        y_test_array: ndarray = np.array(y_test)
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test_array == i, y_scores[:, i])
        average_precision[i] = auc(recall_dict[i], precision_dict[i])

        plt.plot(
            recall_dict[i],
            precision_dict[i],
            lw=2,
            label=f'Class {i} (AUC = {average_precision[i]:.2f})',
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Multi-Class Classification')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    print("Creating a confusion matrix...")
    conf_matrix: ndarray = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.close()

    print("Examining misclassifications...")
    print(f'# sentences: {len(X_test)}')
    print(f'# predicted labels: {len(y_pred)}')
    print(f'# actual labels: {len(y_test)}')

    test_df: DataFrame = DataFrame({
        'text': X_test,
        'predicted_label': y_pred,
        'actual_label': y_test,
    })
    misclassifications = test_df[test_df['predicted_label'] != test_df['actual_label']].reset_index(drop=True)
    predicted_label: str = input("Please enter the predicted label (0-5): ")
    actual_label: str = input("Please enter the actual label (0-5): ")
    examine_misclassifications(df=misclassifications, predicted_label=predicted_label, actual_label=actual_label)
