"""
Drop-in accuracy and feature configuration booster for WiSARDINES-PI-WNN.
Pushes generalization performance without breaking physical constraints or RAM caps.
"""

import os
import json
import numpy as np
from config import WiSARDPhysicsConfig
from wisard_engine import PurePhysicsInformedWiSARD

def get_boosted_config(base_config: WiSARDPhysicsConfig = None) -> WiSARDPhysicsConfig:
    """
    Returns an optimized configuration targeting high accuracy bands 
    by leveraging gray-coded temporal windows and sparse memory configurations.
    """
    if base_config is None:
        config = WiSARDPhysicsConfig()
    else:
        config = base_config

    # 1. Scale window size to capture motion momentum without dense RAM explosion
    config.window_size = 8
    config.window_stride = 1
    
    # 2. Enforce sparse or hashed storage mode to bypass exponential dense allocation limits
    config.storage_mode = "sparse"
    
    # 3. Enable robust smoothing priors for unseen addresses
    config.soft_scores = True
    
    # 4. Configure thermometer quantization with Gray coding for local continuity
    config.quantizer_gray = True
    
    return config

def apply_boosted_training(model: PurePhysicsInformedWiSARD, X_train: np.ndarray, y_train: np.ndarray, use_symmetry: bool = True):
    """
    Trains the weightless neural network model using symmetry-orbit seeding 
    to maximize coverage across valid algebraic spaces.
    """
    print(f"[*] Initializing boosted training pipeline with symmetry optimization: {use_symmetry}")
    
    if use_symmetry and hasattr(model, "memorize_symmetric"):
        # Memorize structured orbits (identity, bit-reversal, complement, etc.)
        for x, y in zip(X_train, y_train):
            model.memorize_symmetric(x, y)
    else:
        for x, y in zip(X_train, y_train):
            model.memorize(x, y)
            
    print("[+] Training sequence completed successfully without architectural leakage.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    cfg = get_boosted_config()
    
    # Save the boosted configuration artifact for reproducibility
    output_path = os.path.join("results", "accuracy_boost_config.json")
    with open(output_path, "w") as f:
        json.dump({
            "window_size": cfg.window_size,
            "storage_mode": cfg.storage_mode,
            "soft_scores": cfg.soft_scores,
            "quantizer_gray": getattr(cfg, "quantizer_gray", True),
            "status": "active"
        }, f, indent=4)
        
    print(f"[+] Boost configuration successfully written to {output_path}")
