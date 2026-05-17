lstm_hyperparameters = {
    'learning_rate': [1e-3, 5e-4],
    'batch_size': [16, 32],
    'weight_decay': [1e-4, 1e-5],
    'dropout': [0.3, 0.5],
    'optimizer': ['Adam', 'AdamW'],
    'epochs': [50],
    'patience': [5]
}


transformer_hyperparameters = {
    'learning_rate': [2e-5, 3e-5, 5e-5],   # Fine-tuning range for BERT encoders
    'batch_size': [16, 32],
    'weight_decay': [0.01, 0.005],
    'optimizer': ['Adam', 'AdamW'],
    'epochs': [20],
    'patience': [5],
    'freeze_layers': [6]                    # Freeze bottom N encoder layers (0 = none, 6 = half)
}