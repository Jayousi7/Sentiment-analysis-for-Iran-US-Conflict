import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from Dataset import TweetDataset, LABEL_MAP
from models import SentimentBiLSTM, get_transformer_model


MODEL_REGISTRY = {
    'arabert': 'aubmindlab/bert-base-arabertv02-twitter',
    'marbert': 'UBC-NLP/MARBERT',
    'bilstm': 'aubmindlab/bert-base-arabertv02-twitter'
}

LABEL_NAMES = ['Negative', 'Neutral', 'Positive']


def load_model(model_type, weights_path, hyperparameters, device):
    if model_type in ['arabert', 'marbert']:
        model = get_transformer_model(MODEL_REGISTRY[model_type], num_labels=3)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY['bilstm'])
        model = SentimentBiLSTM(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=3,
            dropout=hyperparameters.get('dropout', 0.3),
            pad_idx=tokenizer.pad_token_id,
            n_layers=2,
            bidirectional=True
        )

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_on_test(model, test_loader, device, is_transformer):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            if is_transformer:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                text_lengths = batch['text_length']
                logits = model(input_ids, text_lengths)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(true_labels, predictions, model_name, save_dir):
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()


def discover_trained_models(base_path='.'):
    found = {}
    for name in ['arabert', 'marbert', 'bilstm']:
        results_dir = os.path.join(base_path, f'{name}_results', 'best_model')
        weights_path = os.path.join(results_dir, 'weights.pt')
        json_path = os.path.join(results_dir, 'best_results.json')

        if os.path.exists(weights_path) and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                summary = json.load(f)
            found[name] = {
                'weights_path': weights_path,
                'results_dir': results_dir,
                'hyperparameters': summary.get('hyperparameters', {}),
                'val_metrics': summary.get('metrics', {})
            }
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    models_found = discover_trained_models()
    if not models_found:
        print('No trained models found. Run train.py first.')
        return

    print(f'Found {len(models_found)} trained model(s): {list(models_found.keys())}')

    comparison_rows = []

    for model_name, info in models_found.items():
        print(f'\n{"=" * 60}')
        print(f'  Evaluating: {model_name.upper()}')
        print(f'{"=" * 60}')

        is_transformer = model_name in ['arabert', 'marbert']
        tokenizer_name = MODEL_REGISTRY[model_name]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        test_df = pd.read_csv(args.test_path)
        test_dataset = TweetDataset(
            test_df['Tweet Content'].values,
            test_df['Sentiment Class'].values,
            tokenizer,
            args.max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = load_model(model_name, info['weights_path'], info['hyperparameters'], device)
        true_labels, predictions = evaluate_on_test(model, test_loader, device, is_transformer)

        report = classification_report(
            true_labels, predictions,
            target_names=LABEL_NAMES,
            labels=[0, 1, 2],
            output_dict=True,
            zero_division=0
        )

        print(classification_report(
            true_labels, predictions,
            target_names=LABEL_NAMES,
            labels=[0, 1, 2],
            zero_division=0
        ))

        save_confusion_matrix(true_labels, predictions, model_name, output_dir)

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, f'{model_name}_classification_report.csv'))

        comparison_rows.append({
            'Model': model_name.upper(),
            'Accuracy': round(report['accuracy'], 4),
            'Macro F1': round(report['macro avg']['f1-score'], 4),
            'Macro Precision': round(report['macro avg']['precision'], 4),
            'Macro Recall': round(report['macro avg']['recall'], 4),
            'Neg F1': round(report['Negative']['f1-score'], 4),
            'Neutral F1': round(report['Neutral']['f1-score'], 4),
            'Pos F1': round(report['Positive']['f1-score'], 4),
        })

        del model
        torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('Macro F1', ascending=False).reset_index(drop=True)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    print(f'\n{"=" * 60}')
    print('  MODEL COMPARISON (ranked by Macro F1)')
    print(f'{"=" * 60}')
    print(comparison_df.to_string(index=False))
    print(f'\nAll results saved to: {output_dir}/')


if __name__ == '__main__':
    main()
