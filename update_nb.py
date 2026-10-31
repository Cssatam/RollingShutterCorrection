import json

nb_path = r"C:\Users\udayb\.gemini\antigravity\scratch\rolling_shutter_correction\Rolling_Shutter_Correction_Complete_Project_REVISED.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_10_lines = [
    'DATA_DIR = "D:/RS-Real/train"  # Pointing to downloaded RS-Real dataset\n',
    'SAVE_DIR = "./checkpoints"\n',
    '\n',
    'FLOW_SIZE = 64\n',
    'IMAGE_SIZE = 256\n',
    'BATCH_SIZE = 4\n',
    'EPOCHS = 10\n',
    'LEARNING_RATE = 2e-4\n',
    'WEIGHT_DECAY = 1e-4\n',
    'PERCEPTUAL_WEIGHT = 0.1\n',
    'NUM_WORKERS = 0\n',
    '\n',
    'os.makedirs(SAVE_DIR, exist_ok=True)\n',
    'print("Dataset:", DATA_DIR)\n',
    'print("Checkpoints:", SAVE_DIR)\n'
]

nb["cells"][10]["source"] = cell_10_lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Updated notebook config cell!")
