import argparse
import random
import time
import numpy as np
import torch
from torch.utils.data import Subset

from Dataset import get_dataloaders
from models import SentimentBiLSTM, get_transformer_model
from utils import calculate_metrics
from configs import lstm_hyperparameters, transformer_hyperparameters
from grid_search import run_grid_search


def train_model(model, train_loader, val_loader, optimizer, criterion, device, patience=5, num_epochs=None, is_transformer=True):
    if num_epochs is None:
        num_epochs = 20 if is_transformer else 50

    model = model.to(device)

    train_losses, val_losses = [], []
    epoch_metrics_list = []

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0
        all_train_preds, all_train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            if is_transformer:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                text_lengths = batch['text_length']
                logits = model(input_ids, text_lengths)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_metrics = calculate_metrics(all_train_labels, all_train_preds, prefix='train')

        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                if is_transformer:
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    text_lengths = batch['text_length']
                    logits = model(input_ids, text_lengths)

                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_metrics = calculate_metrics(all_val_labels, all_val_preds)

        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics_list.append(epoch_metrics)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Time: {elapsed:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_metrics['val_macro_f1']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    return train_losses, val_losses, epoch_metrics_list



def freeze_transformer_layers(model, num_layers_to_freeze):

    if num_layers_to_freeze <= 0:
        return

    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    for layer_idx in range(num_layers_to_freeze):
        for param in model.base_model.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"  [Freeze] Frozen embeddings + {num_layers_to_freeze} encoder layers "
          f"({frozen}/{total} param tensors frozen)")


def build_model_fn(model_type, config, vocab_size=None, pad_idx=None):
    if model_type == 'bilstm':
        return SentimentBiLSTM(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=3,
            dropout=config['dropout'],
            pad_idx=pad_idx,
            n_layers=2,
            bidirectional=True
        )
    elif model_type in ('arabert', 'marbert'):
        model_name = (
            "aubmindlab/bert-base-arabertv02-twitter"
            if model_type == 'arabert'
            else "UBC-NLP/MARBERT"
        )
        model = get_transformer_model(model_name, num_labels=3)
        freeze_transformer_layers(model, config.get('freeze_layers', 0))
        return model
    else:
        raise ValueError("Invalid model type.")


def apply_dry_run(dataloader):
    subset_indices = list(range(min(64, len(dataloader.dataset))))
    subset = Subset(dataloader.dataset, subset_indices)
    return torch.utils.data.DataLoader(subset, batch_size=dataloader.batch_size,
                                       sampler=None, shuffle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Classification Pipeline")
    parser.add_argument('--model', type=str, required=True, choices=['bilstm', 'arabert', 'marbert'],
                        help='Model to train')
    parser.add_argument('--dry_run', action='store_true', help='Run a quick 2-epoch test on a tiny subset of data')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (defaults: 10 for transformers, 40 for BiLSTM)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    is_transformer = args.model in ['arabert', 'marbert']
    grid = transformer_hyperparameters if is_transformer else lstm_hyperparameters

    epochs_to_use = args.epochs
    if args.dry_run:
        print("\n[WARNING] DRY RUN MODE ENABLED. Training on minimal data for 2 epochs to test pipeline.\n")
        grid = {k: [v[0]] for k, v in grid.items()}
        epochs_to_use = 2

    from transformers import AutoTokenizer

    TOKENIZER_MAP = {
        'arabert': 'aubmindlab/bert-base-arabertv02-twitter',
        'marbert': 'UBC-NLP/MARBERT',
        'bilstm': 'aubmindlab/bert-base-arabertv02-twitter'
    }

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[args.model])


    def get_loaders_for_grid(batch_size):
        train_dl, val_dl, test_dl = get_dataloaders(
            train_path='data/train.csv',
            val_path='data/val.csv',
            test_path='data/test.csv',
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=128
        )
        if args.dry_run:
            train_dl = apply_dry_run(train_dl)
            val_dl = apply_dry_run(val_dl)
        return train_dl, val_dl, test_dl


    vocab_size, pad_idx = None, None
    if args.model == 'bilstm':
        vocab_size = tokenizer.vocab_size
        pad_idx = tokenizer.pad_token_id


    def model_init_wrapper(config):
        return build_model_fn(args.model, config, vocab_size, pad_idx)

    try:
        run_grid_search(
            model_name=args.model + ("_dryrun" if args.dry_run else ""),
            param_grid=grid,
            get_model_fn=model_init_wrapper,
            train_dataloaders_fn=get_loaders_for_grid,
            train_model_fn=train_model,
            device=device,
            is_transformer=is_transformer,
            num_epochs=epochs_to_use
        )
        print("\nPipeline execution finished successfully.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline crashed: {e}")