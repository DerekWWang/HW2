import csv
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from generator import generate_batch, generate_add_sub, generate_mod_div

def load_strings_from_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        lines = [row[0] for row in reader if row]  # avoid empty rows
    return lines

def save_strings_to_csv(lines, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow([line])

def split_and_save(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, out_dir="data", prefix="split"):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_strings_to_csv(train_data, f"{out_dir}/{prefix}_train.csv")
    save_strings_to_csv(val_data, f"{out_dir}/{prefix}_val.csv")
    save_strings_to_csv(test_data, f"{out_dir}/{prefix}_test.csv")

    print(f"Saved {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test samples to `{out_dir}/`")

def concat_csv_strings(filepath):
    full_string = ""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            full_string += ' '.join(row) + ' '
    return full_string.strip()

script_dir = Path(__file__).resolve().parent

grok_path = script_dir / "grok_97.csv"
add_sub_97_path = script_dir / "add_sub_97.csv"
add_sub_113_path = script_dir / "add_sub_113.csv"

raw_data_grok = generate_mod_div(d=5000, p=97, filepath=grok_path)
raw_data_97 = generate_add_sub(d=5000, p=97, filepath=add_sub_97_path)
raw_data_113 = generate_add_sub(d=5000, p=113, filepath=add_sub_113_path)

raw_data_97 = load_strings_from_csv(add_sub_97_path)
raw_data_113 = load_strings_from_csv(add_sub_113_path)
raw_data_grok = load_strings_from_csv(grok_path)

splits_dir = script_dir / "splits"

# ======== Configurables ========
split_and_save(
    data=raw_data_97,
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    out_dir=splits_dir,
    prefix="math_97"
)

split_and_save(
    data=raw_data_113,
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    out_dir=splits_dir,
    prefix="math_113"
)

split_and_save(
    data=raw_data_grok,
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    out_dir=splits_dir,
    prefix="mod_div"
)