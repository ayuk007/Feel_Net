# src/data/dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Special tokens
        self.cls_token = torch.tensor([self.tokenizer.token_to_id("[CLS]")], dtype=torch.int64)
        self.sep_token = torch.tensor([self.tokenizer.token_to_id("[SEP]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        """Get a single item from the dataset."""
        text = self.data.iloc[index]["Text"]
        sentiment = self.data.iloc[index]["Sentiment"]
        
        # Tokenize text
        text_tokens = self.tokenizer.encode(text).ids
        sentiment_token = self.tokenizer.encode(sentiment).ids
        
        # Calculate padding
        text_num_padding_tokens = self.max_len - len(text_tokens) - 2
        
        # Truncate if too long
        if text_num_padding_tokens < 0:
            text_tokens = text_tokens[:self.max_len - 2]
            text_num_padding_tokens = 0
        
        # Create input sequence: [CLS] + tokens + [SEP] + padding
        text_input = torch.cat([
            self.cls_token,
            torch.tensor(text_tokens, dtype=torch.int64),
            self.sep_token,
            torch.tensor([self.pad_token] * text_num_padding_tokens, dtype=torch.int64).flatten(),
        ])
        
        sentiment_input = torch.tensor(sentiment_token, dtype=torch.int64)
        
        return {
            "text_input": text_input,
            "text_input_mask": (text_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "sentiment_input": sentiment_input,
            "text": text,
            "sentiment": sentiment,
        }

def create_data_loaders(train_df: pd.DataFrame, 
                       test_df: pd.DataFrame, 
                       tokenizer, 
                       max_len: int, 
                       batch_size: int,
                       shuffle: bool = True) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    train_dataset = SentimentDataset(train_df, tokenizer, max_len)
    val_dataset = SentimentDataset(test_df, tokenizer, max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    return train_loader, val_loader