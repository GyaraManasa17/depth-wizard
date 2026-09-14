import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import h5py
import numpy as np
from PIL import Image
from transformers import pipeline
from sklearn.linear_model import LinearRegression

GAMUS_DIR = "../data/gamus"
SUBSAMPLE = 8  # only use every 8th pixel (in each direction) - keeps training set manageable

def load_pair(scene, split):
    with h5py.File(f"{GAMUS_DIR}/images/{split}/{scene}_RGB.h5", "r") as f:
        rgb = f["image"][:]
    with h5py.File(f"{GAMUS_DIR}/heights/{split}/{scene}_AGL.h5", "r") as f:
        height = f["image"][:]
    return rgb, height

def get_scenes(split):
    img_dir = f"{GAMUS_DIR}/images/{split}"
    return sorted(f.replace("_RGB.h5", "") for f in os.listdir(img_dir) if f.endswith(".h5"))

print("Loading depth model...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

def extract_pixel_pairs(scenes, split):
    all_depth, all_height = [], []
    for scene in scenes:
        print(f"  processing {scene}...")
        rgb, height = load_pair(scene, split)
        image = Image.fromarray(rgb)
        result = pipe(image)
        relative_depth = np.array(result["depth"]).astype(np.float64)

        # Subsample to keep things fast and avoid massively over-representing any one scene
        d_sub = relative_depth[::SUBSAMPLE, ::SUBSAMPLE].ravel()
        h_sub = height[::SUBSAMPLE, ::SUBSAMPLE].ravel()

        valid = ~np.isnan(h_sub)
        all_depth.append(d_sub[valid])
        all_height.append(h_sub[valid])
    return np.concatenate(all_depth), np.concatenate(all_height)

print("\nExtracting training pixel pairs...")
train_scenes = get_scenes("train")
train_depth, train_height = extract_pixel_pairs(train_scenes, "train")
print(f"Total training pixels: {len(train_depth)}")

print("\nExtracting validation pixel pairs...")
val_scenes = get_scenes("val")
val_depth, val_height = extract_pixel_pairs(val_scenes, "val")
print(f"Total validation pixels: {len(val_depth)}")

print("\nFitting linear regression (trained across ALL 30 scenes, not just one)...")
model = LinearRegression()
model.fit(train_depth.reshape(-1, 1), train_height)
print(f"Fit: real_height = {model.coef_[0]:.4f} * relative_depth + {model.intercept_:.4f}")

# Evaluate on held-out VAL SCENES the model never saw during training
val_predictions = model.predict(val_depth.reshape(-1, 1))
residuals = val_predictions - val_height
rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))
correlation = np.corrcoef(val_depth, val_height)[0, 1]

print(f"\n=== HELD-OUT VALIDATION (10 scenes never used in training) ===")
print(f"RMSE: {rmse:.2f} m")
print(f"MAE: {mae:.2f} m")
print(f"Correlation: {correlation:.3f}")

import pickle
with open("../backend/gamus_calibration_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nSaved trained calibration model to backend/gamus_calibration_model.pkl")