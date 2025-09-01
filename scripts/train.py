import pandas as pd
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import Tuple, List
from torch.utils.data import DataLoader

from src.models.model import FeelNet
from config import load_config, Config
from src.data import create_data_loaders
from src.training.trainer import Trainer


def setup_tokenizer(config_: Config) -> Tokenizer:
    """
    Initialize and train a BPE tokenizer with specified special tokens.
    
    Args:
        config_ (Config): Configuration object containing data file paths
        
    Returns:
        Tokenizer: Trained BPE tokenizer
    """
    special_tokens = [
        "Neutral", "Positive", "Negative", "Irrelevant",
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"
    ]
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=special_tokens)
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train tokenizer on both train and test files
    files = [config_.data.train_file, config_.data.test_file]
    tokenizer.train(files)
    
    return tokenizer


def create_model(config_: Config) -> FeelNet:
    """
    Initialize the FeelNet model with configuration parameters.
    
    Args:
        config_ (Config): Configuration object containing model parameters
        
    Returns:
        FeelNet: Initialized model instance
    """
    model = FeelNet(
        d_model=config_.model.d_model,
        num_heads=config_.model.num_heads,
        dropout=config_.model.dropout,
        n_layers=config_.model.n_layers,
        vocab_size=config_.model.vocab_size,
        num_classes=config_.n_classes,
        seq_len=config_.model.max_seq_len
    )
    return model


def get_data(config_: Config) -> Tuple[DataLoader, DataLoader, Tokenizer]:
    """
    Prepare data loaders and tokenizer for training.
    
    Args:
        config_ (Config): Configuration object containing data and model parameters
        
    Returns:
        Tuple containing:
            - train_dataloader (DataLoader): DataLoader for training data
            - val_dataloader (DataLoader): DataLoader for validation data
            - tokenizer (Tokenizer): Trained tokenizer
    """
    tokenizer = setup_tokenizer(config_)
    
    # Load and prepare datasets
    train_df = pd.read_csv(config_.data.train_file)
    test_df = pd.read_csv(config_.data.test_file)
    
    train_dataloader, val_dataloader = create_data_loaders(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_len=config_.model.max_seq_len,
        batch_size=config_.training.batch_size,
    )
    
    return train_dataloader, val_dataloader, tokenizer


def setup_trainer(
    model: FeelNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Tokenizer,
    config_: Config
) -> Trainer:
    """
    Initialize the trainer with model, data loaders, and training parameters.
    
    Args:
        model (FeelNet): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        tokenizer (Tokenizer): Trained tokenizer
        config_ (Config): Configuration object
        
    Returns:
        Trainer: Configured trainer instance
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_.training.learning_rate,
        weight_decay=config_.training.weight_decay
    )
    
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id('[PAD]')
    ).to(device="cuda")
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cuda",
        class_names=['Irrelevant', 'Negative', 'Neutral', 'Positive'],
        save_dir=config_.training.checkpoint_dir,
        log_dir=config_.training.log_dir
    )


def main():
    """
    Main training pipeline.
    """
    # Load configuration
    config_ = load_config()
    
    # Initialize model and prepare data
    model = create_model(config_)
    train_loader, val_loader, tokenizer = get_data(config_)
    
    # Setup and start training
    trainer = setup_trainer(model, train_loader, val_loader, tokenizer, config_)
    trainer.fit(num_epochs=config_.training.num_epochs)


if __name__ == "__main__":
    main()