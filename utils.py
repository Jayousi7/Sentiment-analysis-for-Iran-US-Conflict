import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


def calculate_metrics(true_labels, predictions, prefix='val'):

    acc = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)


    precisions = precision_score(true_labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)
    recalls = recall_score(true_labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        f'{prefix}_accuracy': acc,
        f'{prefix}_macro_f1': macro_f1,
        f'{prefix}_precision_neg': precisions[0],
        f'{prefix}_precision_neutral': precisions[1],
        f'{prefix}_precision_pos': precisions[2],
        f'{prefix}_recall_neg': recalls[0],
        f'{prefix}_recall_neutral': recalls[1],
        f'{prefix}_recall_pos': recalls[2]
    }


def plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, save_dir):

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.plot(epochs, val_f1s, 'm-', label='Validation Macro F1')
    ax2.set_title('Validation Metrics over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()