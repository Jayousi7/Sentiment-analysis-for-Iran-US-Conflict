import os
import csv
import json
import itertools
import shutil
import torch
import torch.nn as nn

from utils import plot_learning_curves


def create_directory_structure(model_name):
    base_dir = f"{model_name}_results"
    best_dir = os.path.join(base_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    return base_dir, best_dir


def run_grid_search(model_name, param_grid, get_model_fn, train_dataloaders_fn, train_model_fn, device,
                    is_transformer=True, num_epochs=None):

    base_dir, best_dir = create_directory_structure(model_name)
    csv_file_path = os.path.join(base_dir, "all_configs_results.csv")

    keys, values = zip(*param_grid.items())
    config_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_overall_f1 = 0.0

    metric_keys = [
        'train_macro_f1', 'val_macro_f1',
        'train_accuracy', 'val_accuracy',
        'train_precision_neg', 'train_precision_neutral', 'train_precision_pos',
        'train_recall_neg', 'train_recall_neutral', 'train_recall_pos',
        'val_precision_neg', 'val_precision_neutral', 'val_precision_pos',
        'val_recall_neg', 'val_recall_neutral', 'val_recall_pos'
    ]

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = list(keys) + metric_keys + ['stopped_epoch']
        writer.writerow(header)
        file.flush()

        for idx, config in enumerate(config_combinations):
            print(f"\n--- Config {idx + 1}/{len(config_combinations)}: {config} ---")

            train_loader, val_loader, test_loader = train_dataloaders_fn(batch_size=config['batch_size'])
            model = get_model_fn(config)

            optimizer_class = getattr(torch.optim, config['optimizer'])
            optimizer = optimizer_class(model.parameters(),
                                        lr=config['learning_rate'],
                                        weight_decay=config['weight_decay'])

            epochs_to_use = num_epochs if num_epochs is not None else config.get('epochs')
            patience_to_use = config.get('patience', 3)

            criterion = nn.CrossEntropyLoss()

            train_losses, val_losses, epoch_metrics_list = train_model_fn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                is_transformer=is_transformer,
                num_epochs=epochs_to_use,
                patience=patience_to_use
            )

            best_epoch_idx = val_losses.index(min(val_losses))
            best_epoch_metrics = epoch_metrics_list[best_epoch_idx]

            metric_values = [best_epoch_metrics.get(k, 0.0) for k in metric_keys]
            stopped_epoch = best_epoch_idx + 1

            row = list(config.values()) + metric_values + [stopped_epoch]
            writer.writerow(row)
            file.flush()

            if best_epoch_metrics['val_macro_f1'] > best_overall_f1:
                best_overall_f1 = best_epoch_metrics['val_macro_f1']

                if os.path.exists('best_model.pt'):
                    shutil.move('best_model.pt', os.path.join(best_dir, 'weights.pt'))

                val_acc_list = [m['val_accuracy'] for m in epoch_metrics_list]
                val_f1_list = [m['val_macro_f1'] for m in epoch_metrics_list]

                plot_learning_curves(train_losses, val_losses, val_acc_list, val_f1_list, save_dir=best_dir)

                summary = {
                    'hyperparameters': config,
                    'stopped_at_epoch': stopped_epoch,
                    'metrics': best_epoch_metrics
                }
                with open(os.path.join(best_dir, 'best_results.json'), 'w') as json_file:
                    json.dump(summary, json_file, indent=4)

            if os.path.exists('best_model.pt'):
                os.remove('best_model.pt')