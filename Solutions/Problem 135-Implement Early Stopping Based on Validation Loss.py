# https://www.deep-ml.com/problems/135

from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_epoch=0
    best_loss = val_losses[0]
    patience_counter=0
    for epoch in range(1,len(val_losses)):
        if val_losses[epoch] < best_loss - min_delta:
            best_loss = val_losses[epoch]
            best_epoch = epoch
            patience_counter = 0
        else :
            patience_counter += 1
        if patience_counter >= patience:
            return epoch, best_epoch
    return len(val_losses)-1, best_epoch
