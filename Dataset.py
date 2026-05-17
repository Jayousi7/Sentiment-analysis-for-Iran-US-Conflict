import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


LABEL_MAP = {'neg': 0, 'neutral': 1, 'pos': 2}


class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = LABEL_MAP

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label_str = str(self.labels[idx]).strip().lower()
        label_idx = self.label_map[label_str]

        encoding = self.tokenizer(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'text_length': (encoding['input_ids'].flatten() != self.tokenizer.pad_token_id).sum()
        }


def get_dataloaders(train_path: str, val_path: str, test_path: str, tokenizer, batch_size=16, max_length=128):

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_texts, train_labels = train_df['Tweet Content'].values, train_df['Sentiment Class'].values
    val_texts, val_labels = val_df['Tweet Content'].values, val_df['Sentiment Class'].values
    test_texts, test_labels = test_df['Tweet Content'].values, test_df['Sentiment Class'].values

    train_dataset = TweetDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TweetDataset(test_texts, test_labels, tokenizer, max_length)

    label_map = LABEL_MAP
    train_labels_int = [label_map[str(label).strip().lower()] for label in train_labels]

    class_counts = np.bincount(train_labels_int)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels_int]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")

    train_dl, val_dl, test_dl = get_dataloaders(
        train_path='data/train.csv',
        val_path='data/val.csv',
        test_path='data/test.csv',
        tokenizer=arabert_tokenizer,
        batch_size=32,
        max_length=256
    )

    print(f"Training batches: {len(train_dl)}")
    print("Data loading pipeline ready!")