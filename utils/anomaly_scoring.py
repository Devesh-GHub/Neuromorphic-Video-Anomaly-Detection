import torch
import numpy as np


def compute_anomaly_scores(model, dataloader, device):
    model.eval()
    scores = []


    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            # ⚡ Mixed precision forward
            with torch.amp.autocast('cuda'):
                reconstructed = model(batch)

            # ⚡ Compute error entirely on GPU (no .cpu() yet)
            errors = ((batch - reconstructed) ** 2).mean(dim=(1,2,3,4))

            # Move to CPU only once per batch
            scores.extend(errors.detach().cpu().numpy())

            # Progress print (every 20 batches)
            if batch_idx % 20 == 0:
                print(f"[Evaluation] {batch_idx}/{len(dataloader)} batches processed...")

    return np.array(scores)



def normalize_scores(scores):
    """
    Normalize anomaly scores to [0, 1].

    Args:
        scores (np.ndarray)

    Returns:
        np.ndarray
    """
    if len(scores) == 0:
        return scores
    
    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val == 0:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)
