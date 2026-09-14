from huggingface_hub import HfApi, hf_hub_download
import os

REPO_ID = "earthflow/GAMUS"
REPO_TYPE = "dataset"
OUTPUT_DIR = "../data/gamus"
N_TRAIN = 30
N_VAL = 10

api = HfApi()
print("Listing files in the dataset repo...")
all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)

def get_scene_names(files, split, prefix):
    # e.g. images/train/DC_04_23_RGB.h5 -> "DC_04_23"
    matches = [f for f in files if f.startswith(f"{prefix}/{split}/") and f.endswith(".h5")]
    scenes = sorted(set(f.split("/")[-1].replace("_RGB.h5", "").replace("_AGL.h5", "") for f in matches))
    return scenes

def download_pairs(split, n_pairs):
    image_scenes = set(get_scene_names(all_files, split, "images"))
    height_scenes = set(get_scene_names(all_files, split, "heights"))
    matched_scenes = sorted(image_scenes & height_scenes)  # only scenes with BOTH files present
    print(f"{split}: {len(matched_scenes)} matched scenes available, downloading {n_pairs}")

    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/heights/{split}", exist_ok=True)

    downloaded = []
    for scene in matched_scenes[:n_pairs]:
        rgb_path = hf_hub_download(REPO_ID, f"images/{split}/{scene}_RGB.h5", repo_type=REPO_TYPE,
                                     local_dir=OUTPUT_DIR)
        height_path = hf_hub_download(REPO_ID, f"heights/{split}/{scene}_AGL.h5", repo_type=REPO_TYPE,
                                        local_dir=OUTPUT_DIR)
        downloaded.append(scene)
        print(f"  downloaded {scene}")
    return downloaded

train_scenes = download_pairs("train", N_TRAIN)
val_scenes = download_pairs("val", N_VAL)

print(f"\nDone. {len(train_scenes)} train pairs, {len(val_scenes)} val pairs.")