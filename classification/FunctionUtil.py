from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def oversample_data(x_reviews, y_labels):
    oversampler = RandomOverSampler()
    return oversampler.fit_resample(x_reviews, y_labels)

def undersample_data(x_reviews, y_labels):
    undersampler = RandomUnderSampler()
    return undersampler.fit_resample(x_reviews, y_labels)

def plot_model_comparisons(results_list):
    metrics = ['precision', 'recall', 'f1']
    classes = ['Petit', 'Moyen','Long']
    n_models = len(results_list)
    model_names = [result['model'] for result in results_list]

    # Plot for each class and each metric
    for class_index in range(2):
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle(f'Model Performance Comparison for {classes[class_index]} - {metric.capitalize()}')

            values = [result[metric][class_index] for result in results_list]
            bars = ax.bar(np.arange(n_models), values, align='center', alpha=0.7)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

            ax.set_xticks(np.arange(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            plt.tight_layout()
            plt.show()

    # Separate plot for weighted F1 score
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Model Performance Comparison - Weighted F1 Score')
    weighted_f1_values = [result['weighted_f1'] for result in results_list]

    bars = ax.bar(np.arange(n_models), weighted_f1_values, align='center', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')
    
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel('Weighted F1 Score')
    ax.set_title('Weighted F1 Score Comparison')
    plt.tight_layout()
    plt.show()

def metrics(y_test, y_pred, model_name,results_list):
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Petit','Moyen','Long']).plot(values_format='d')
    plt.show()

    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("Precision per class:\n", precision_per_class)
    print("Recall per class:\n", recall_per_class)
    print("F1 Score per class:\n", f1_per_class)
    print("Weighted F1 Score:", weighted_f1)

    # Append the results to the global list
    results_list.append({
        'model': model_name,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1': f1_per_class,
        'weighted_f1': weighted_f1
    })

