# train_emotion_csv.py
"""
Usage:
    python train_emotion_csv.py
This script attempts to parse fer2013_training_onehot.csv (or similar)
placed in the same folder. It detects common formats:
 - a 'pixels' column with "0 0 0 ... 255" strings (original FER2013)
 - many columns containing pixel values (2304 columns for 48x48)
 - one-hot label columns or a single 'emotion'/'label' column

Outputs:
 - prints diagnostics
 - saves X.npy (N,48,48) and y.npy (N,) in ./processed_data/
 - optionally writes images to ./processed_data/images/{label}/ (toggle below)
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

CSV = "fer2013_training_onehot.csv"  # change if needed
OUTDIR = "processed_data"
SAVE_IMAGES = False   # set True to dump images into folders (slow)
IMAGE_SIZE = (48, 48)
SCALE_TO_0_1 = True   # normalize to 0..1

def detect_and_load(csv_path):
    print("Loading CSV (only first 5 rows for quick preview)...")
    try:
        df_pre = pd.read_csv(csv_path, nrows=5)
    except Exception as e:
        print("Failed loading CSV preview:", e)
        raise
    print("Columns preview:", df_pre.columns.tolist()[:50])
    # Load full file (pandas will infer types)
    print("Loading full CSV (this may take a few seconds)...")
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    return df

def find_pixels_array(df):
    # Case A: 'pixels' column as string of space-separated ints
    if 'pixels' in df.columns:
        print("Detected 'pixels' column (space-separated). Parsing...")
        pixels = df['pixels'].astype(str).tolist()
        N = len(pixels)
        X = np.zeros((N, IMAGE_SIZE[0]*IMAGE_SIZE[1]), dtype=np.uint8)
        for i, s in enumerate(pixels):
            parts = s.strip().split()
            if len(parts) != IMAGE_SIZE[0]*IMAGE_SIZE[1]:
                # try to handle stray commas
                joined = s.replace(',', ' ').strip()
                parts = joined.split()
            if len(parts) != IMAGE_SIZE[0]*IMAGE_SIZE[1]:
                raise ValueError(f"Row {i} pixel count {len(parts)} != {IMAGE_SIZE[0]*IMAGE_SIZE[1]}")
            X[i] = np.array(parts, dtype=np.uint8)
        return X
    # Case B: Many pixel columns (>=2304)
    pixel_cols = [c for c in df.columns if c.lower().startswith('pixel') or c.isdigit() or c.lower().startswith('p')]
    if len(pixel_cols) >= IMAGE_SIZE[0]*IMAGE_SIZE[1]:
        print(f"Detected {len(pixel_cols)} pixel-like columns. Using those.")
        # take first 2304 of them in column order
        cols = pixel_cols[:IMAGE_SIZE[0]*IMAGE_SIZE[1]]
        X = df[cols].values.astype(np.uint8)
        return X
    # Case C: The dataframe contains many numeric columns and no explicit pixel prefix.
    # Heuristic: if number of columns >= 2304 + label columns, assume first 2304 are pixels
    if df.shape[1] >= IMAGE_SIZE[0]*IMAGE_SIZE[1] + 1:
        print("CSV has many columns. Assuming first 2304 columns are pixels.")
        X = df.iloc[:, :IMAGE_SIZE[0]*IMAGE_SIZE[1]].values.astype(np.uint8)
        return X
    # Case D: maybe CSV encoded pixel string in first column (rare)
    first_col = df.columns[0]
    if df[first_col].dtype == object:
        sample = str(df[first_col].iloc[0])
        if len(sample.split()) >= IMAGE_SIZE[0]*IMAGE_SIZE[1]:
            print(f"Detected pixel string in first column '{first_col}'. Parsing.")
            pixels = df[first_col].astype(str).tolist()
            N = len(pixels)
            X = np.zeros((N, IMAGE_SIZE[0]*IMAGE_SIZE[1]), dtype=np.uint8)
            for i, s in enumerate(pixels):
                parts = s.strip().split()
                if len(parts) != IMAGE_SIZE[0]*IMAGE_SIZE[1]:
                    parts = s.replace(',', ' ').split()
                X[i] = np.array(parts, dtype=np.uint8)
            return X
    raise ValueError("Unable to find pixel data automatically. Inspect CSV columns manually.")

def find_labels(df):
    # If there's an 'emotion' or 'label' column return it
    for name in ['emotion','label','labels','emotion_label','target']:
        if name in df.columns:
            print(f"Found label column: {name}")
            col = df[name]
            # if column is numeric 0..6 etc, return it
            if pd.api.types.is_numeric_dtype(col):
                return col.values.astype(int)
            # otherwise try to map strings to ints
            uniques = sorted(col.unique())
            mapping = {v:i for i,v in enumerate(uniques)}
            print("Mapping label strings -> ints:", mapping)
            return col.map(mapping).values.astype(int)
    # One-hot detection: find columns that look like one-hot (0/1 and row-sums ~1)
    candidate_cols = []
    for c in df.columns:
        # ignore obvious pixel columns
        if c.lower()=='pixels': continue
        # pick columns with only 0/1 values (or 0.0/1.0)
        s = df[c].dropna()
        if s.size == 0: continue
        unique_vals = set(np.unique(s.values))
        # allow floats like 0.0/1.0
        if unique_vals.issubset({0,1,0.0,1.0}):
            candidate_cols.append(c)
    if candidate_cols:
        print("Found candidate one-hot columns:", candidate_cols[:20], " (using all of them)")
        onehot = df[candidate_cols].values.astype(int)
        row_sums = onehot.sum(axis=1)
        # some CSVs have extra columns; select subset where row_sums==1 majority
        if (row_sums==1).mean() < 0.5:
            # try to pick contiguous block of candidate columns where row_sum==1
            print("Warning: fewer than 50% rows had exactly one '1' in these columns.")
        labels = np.argmax(onehot, axis=1)
        return labels
    # As a last resort, check final column for small-integer values
    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]):
        print(f"Using last column '{last}' as labels (numeric).")
        return df[last].values.astype(int)
    raise ValueError("Unable to find labels automatically. Inspect CSV columns manually.")

def main():
    csv_path = CSV
    if not os.path.exists(csv_path):
        print("CSV not found at", csv_path)
        sys.exit(1)
    df = detect_and_load(csv_path)

    X_flat = find_pixels_array(df)  # shape (N, 2304)
    print("Pixels loaded. shape =", X_flat.shape)
    try:
        y = find_labels(df)
        print("Labels loaded. shape =", y.shape, ", unique labels:", np.unique(y))
    except Exception as e:
        print("Label detection failed:", e)
        y = None

    N = X_flat.shape[0]
    # reshape into (N, H, W)
    X = X_flat.reshape((N, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    if SCALE_TO_0_1:
        X = (X.astype(np.float32) / 255.0).astype(np.float32)

    os.makedirs(OUTDIR, exist_ok=True)
    np.save(os.path.join(OUTDIR, "X.npy"), X)
    if y is not None:
        np.save(os.path.join(OUTDIR, "y.npy"), y)
    print("Saved X.npy and y.npy (if labels found) in", OUTDIR)

    if SAVE_IMAGES and y is not None:
        print("Saving images into folders by label (this may take long)...")
        img_dir = os.path.join(OUTDIR, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i in tqdm(range(N)):
            lab = str(int(y[i]))
            d = os.path.join(img_dir, lab)
            if not os.path.exists(d):
                os.makedirs(d)
            arr = (X[i]*255).astype(np.uint8) if SCALE_TO_0_1 else X[i].astype(np.uint8)
            im = Image.fromarray(arr)
            im = im.resize(IMAGE_SIZE)
            im.save(os.path.join(d, f"{i}.png"))
        print("Images saved to", img_dir)

    print("Done. Sample summary:")
    print("  X dtype:", X.dtype, "min/max:", X.min(), X.max())
    if y is not None:
        print("  y dtype:", y.dtype, "unique labels:", np.unique(y), "counts:", {int(k):int((y==k).sum()) for k in np.unique(y)})

if __name__ == "__main__":
    main()
