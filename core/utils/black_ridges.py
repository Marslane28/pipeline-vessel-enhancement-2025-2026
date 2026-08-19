import numpy as np


def detect_black_ridges(image, ground_truth=None, sample_size=10000):
    """
    Détecte automatiquement si black_ridges doit être True ou False, selon
    la convention scikit-image (black_ridges=True → le filtre cherche des
    structures SOMBRES sur fond clair ; False → structures CLAIRES sur fond
    sombre).
    
    Compare l'intensité moyenne des voxels vasculaires (ground_truth > 0) 
    à l'intensité moyenne du reste de l'image. Si les vaisseaux sont plus
    clairs que le reste (cas typique CT avec contraste), on veut détecter
    des crêtes claires → black_ridges=False. Si les vaisseaux sont plus
    sombres, black_ridges=True.

    """
    if ground_truth is None:
        return False, {"error": "Aucun ground truth fourni, black_ridges=False par défaut- Attention: on retourne False par défaut, car nos trois datasets les vaisseaux sont clairs sur fond sombre."}

    if ground_truth.dtype == bool:
        vessel_binary = ground_truth
    elif ground_truth.dtype == np.uint8 and np.all((ground_truth == 0) | (ground_truth == 255)):
        vessel_binary = ground_truth > 127
    else:
        vessel_binary = ground_truth > 0

    inside_idx = np.where(vessel_binary)
    n_in = len(inside_idx[0])
    if n_in == 0:
        return False, {"error": "Ground truth vide (aucun vaisseau détecté)"}
    if n_in > sample_size:
        idx = np.random.choice(n_in, sample_size, replace=False)
        inside_idx = tuple(coord[idx] for coord in inside_idx)

    outside_idx = np.where(~vessel_binary)
    n_out = len(outside_idx[0])
    if n_out > sample_size:
        idx = np.random.choice(n_out, sample_size, replace=False)
        outside_idx = tuple(coord[idx] for coord in outside_idx)

    inside_mean = np.mean(image[inside_idx])
    outside_mean = np.mean(image[outside_idx])

    return inside_mean < outside_mean, {
        "inside_mean": float(inside_mean),
        "outside_mean": float(outside_mean),
        "difference": float(inside_mean - outside_mean),
    }