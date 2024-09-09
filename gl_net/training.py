import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from torchinfo import summary
from tqdm import tqdm

from gl_net import IODataset, GLNetMLP


@torch.no_grad()
def calculate_rmse(predictions: Tensor,
                   targets: Tensor) -> float:
    mse = torch.nn.functional.mse_loss(predictions, targets)
    return torch.sqrt(mse).item()


@torch.no_grad()
def evaluate_model(model: GLNetMLP,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   description: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0

    progress_bar = tqdm(dataloader, total=len(dataloader), desc=description)

    for inputs, targets in progress_bar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        rmse = calculate_rmse(outputs, targets)
        total_rmse += rmse

        # Update tqdm progress bar
        progress_bar.set_postfix({'val_loss': total_loss / (progress_bar.n + 1),
                                  'val_rmse': total_rmse / (progress_bar.n + 1)})

    avg_loss = total_loss / len(dataloader)
    avg_rmse = total_rmse / len(dataloader)
    return {'loss': avg_loss, 'rmse': avg_rmse}


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create dataset and splits
    io_record_csv_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLightPlus/data/io_records/20240819_115758/io_record_step_0.csv"
    dataset = IODataset(io_record_csv_path,
                        rescale_d=True)
    train_size = int(0.9 * len(dataset))  # 90% for training
    val_size = int(0.05 * len(dataset))  # 5% for validation
    test_size = len(dataset) - train_size - val_size  # 5% for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    bsz = 256
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False, num_workers=4)

    model = GLNetMLP(
        input_a_dims=None,
        input_p_dims=None,
        arc_variation=2,
        use_final_tanh=True
    ).to(device)
    summary(model)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training & Validation Loop
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = "glnet_model_best.pth"

    for epoch in range(num_epochs):
        # TRAINING STEP
        model.train()
        running_loss = 0.0
        running_rmse = 0.0

        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, targets) in progress_bar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            rmse = calculate_rmse(outputs, targets)
            running_rmse += rmse

            progress_bar.set_postfix({'step_loss': loss.item(),
                                      'loss': running_loss / (i + 1),
                                      'rmse': running_rmse / (i + 1)})

        epoch_loss = running_loss / len(train_loader)
        epoch_rmse = running_rmse / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train RMSE: {epoch_rmse:.4f}")

        # VALIDATION STEP
        val_metrics = evaluate_model(model,
                                     val_loader,
                                     criterion,
                                     device,
                                     "Validation")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_metrics['loss']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")

        # Early stopping and model saving
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best model and test it
    print("Loading the best model and testing...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate_model(model,
                                  test_loader,
                                  criterion,
                                  device,
                                  "Testing")
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test RMSE: {test_metrics['rmse']:.4f}")
    torch.save(model.state_dict(), "glnet_model_final.pth")
