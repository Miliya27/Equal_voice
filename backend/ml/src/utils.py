import numpy as np

def normalize_vec126(vec126: np.ndarray) -> np.ndarray:
    """
    Beginner-friendly normalization:
    For each hand (63):
      - reshape to (21,3)
      - subtract wrist (landmark 0) => translation invariant
      - scale by max distance from wrist (avoid size issues)
    If hand is all zeros => leave as zeros.
    """
    vec126 = vec126.astype(np.float32).copy()

    def norm_hand(vec63):
        if np.allclose(vec63, 0):
            return vec63
        pts = vec63.reshape(21, 3)
        wrist = pts[0].copy()
        pts = pts - wrist  # shift
        d = np.linalg.norm(pts, axis=1).max()
        if d > 1e-6:
            pts = pts / d
        return pts.reshape(-1)

    left = norm_hand(vec126[:63])
    right = norm_hand(vec126[63:])
    return np.concatenate([left, right]).astype(np.float32)

def majority_vote(items):
    # items: list of labels
    if not items:
        return None
    vals, counts = np.unique(items, return_counts=True)
    return vals[np.argmax(counts)]
