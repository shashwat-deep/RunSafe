import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

class NNConfig(BaseModel):
    input_size: int = 10 # 7 numeric features + 3 XGBoost probabilities
    hidden_layer_size: int = 64
    output_size: int = 3
    dropout_rate: float = 0.4

class PersonalizedInjuryModel(nn.Module):
    def __init__(self, config: NNConfig):
        super(PersonalizedInjuryModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_layer_size),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits for training."""
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Softmax for inference."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

class InjuryModelTrainer:
    def __init__(self, config: NNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _dynamic_min_max_scale(self, features: np.ndarray) -> np.ndarray:
        """Applies N=1 dynamic Min-Max scaling to prevent magnitude dominance."""
        feature_min = features.min(axis=0)
        feature_max = features.max(axis=0)
        # Avoid division by zero for constant features (like 0-load days)
        range_vals = np.where(feature_max - feature_min == 0, 1.0, feature_max - feature_min)
        scaled_features = (features - feature_min) / range_vals
        return scaled_features

    def build_replay_buffer(self, raw_logs: List[Dict], target_date: datetime, window_days: int = 30):
        """
        Interpolates missing days as 0-load rest days and calculates:
        1. Exponential decay sample weights (7-day half-life)
        2. Dynamic class weights with smoothing
        """
        buffer_features, buffer_targets, sample_weights = [], [], []
        class_counts = {0: 0, 1: 0, 2: 0}
        
        # Create a lookup for existing logs
        log_dict = {datetime.fromisoformat(log['date']).date(): log for log in raw_logs}
        
        for i in range(window_days):
            current_date = (target_date - timedelta(days=i)).date()
            
            # 7-day half-life exponential decay formula
            weight = (0.5) ** (i / 7.0)
            sample_weights.append(weight)
            
            if current_date in log_dict:
                log = log_dict[current_date]
                buffer_features.append(log['features']) # Expected: array of 10 floats
                target = log['target_class'] # Expected: 0 (Low), 1 (Med), 2 (High)
                buffer_targets.append(target)
                class_counts[target] += 1
            else:
                # Missing day = 0-load rest day. 
                # Assuming index 0-3 are mileage related, index 7-9 are XGBoost outputs.
                zero_load = [0.0] * self.config.input_size
                zero_load[-3:] = [1.0, 0.0, 0.0] # Default XGBoost to 100% "Low Risk" on rest days
                buffer_features.append(zero_load)
                buffer_targets.append(0) # Target is Low Risk (0)
                class_counts[0] += 1

        # Scale features
        scaled_features = self._dynamic_min_max_scale(np.array(buffer_features))
        
        # Calculate Class Weights with +1 Smoothing
        total_samples = window_days
        class_weights = []
        for i in range(3):
            smoothed_count = class_counts[i] + 1
            class_weights.append(total_samples / (3.0 * smoothed_count))
            
        return (
            torch.FloatTensor(scaled_features).to(self.device),
            torch.LongTensor(buffer_targets).to(self.device),
            torch.FloatTensor(sample_weights).to(self.device),
            torch.FloatTensor(class_weights).to(self.device)
        )

    def fine_tune_n1_model(self, model: nn.Module, raw_logs: List[Dict], target_date: datetime):
        """Executes the N=1 fine-tuning loop."""
        
        X, y, sample_w, class_w = self.build_replay_buffer(raw_logs, target_date)
        model = model.to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # ExponentialLR scheduler to smoothly reduce step size
        scheduler = ExponentialLR(optimizer, gamma=0.9) 
        
        # reduction='none' is CRITICAL so we can multiply by sample_weights
        criterion = nn.CrossEntropyLoss(weight=class_w, reduction='none')
        
        epochs = 20
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            logits = model(X)
            raw_loss = criterion(logits, y)
            
            # Apply 7-day half-life exponential decay weights
            weighted_loss = (raw_loss * sample_w).mean()
            
            weighted_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
        return model
