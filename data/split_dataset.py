"""
Splits the raw Kaggle brain tumor MRI dataset into train/val/test folders.

Important: the raw dataset contains duplicate images saved under different
filenames (confirmed via MD5 content hashing -- 25 duplicate files out of 253
in the 'yes'/'no' folders). Splitting by filename alone, as the previous
version of this script did, lets the *same* image end up in two different
splits (e.g. train and test), which is a form of data leakage: the model can
effectively "see" a test image during training and the reported validation/
test accuracy becomes inflated and not trustworthy.

This version deduplicates by file content (MD5 hash) BEFORE splitting, so
every image that ends up in the final dataset is unique, and no image can
possibly appear in more than one split. It also uses a fixed random seed so
the split is reproducible.
"""

import argparse
import hashlib
import os
import random
import shutil


def md5_of_file(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def dedupe_by_content(files):
    """Given a list of file paths, return only the first file for each
    unique content hash (drops byte-identical duplicates under any
    filename)."""
    seen_hashes = set()
    unique_files = []
    dropped = []
    for f in files:
        digest = md5_of_file(f)
        if digest in seen_hashes:
            dropped.append(f)
            continue
        seen_hashes.add(digest)
        unique_files.append(f)
    return unique_files, dropped


def split_dataset(original_dataset_dir, output_base_dir, seed=42,
                   split_ratios=None):
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    rng = random.Random(seed)
    categories = ['yes', 'no']
    output_dirs = ['train', 'val', 'test']

    for split in output_dirs:
        for category in categories:
            split_dir = os.path.join(output_base_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            # Clear out any previously-split files so re-running this
            # script never leaves stale images mixed in with a new split.
            for existing in os.listdir(split_dir):
                existing_path = os.path.join(split_dir, existing)
                if os.path.isfile(existing_path):
                    os.remove(existing_path)

    summary = {}

    for category in categories:
        category_path = os.path.join(original_dataset_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category folder '{category}' not found!")
            continue

        files = [os.path.join(category_path, f)
                 for f in os.listdir(category_path)
                 if os.path.isfile(os.path.join(category_path, f))]
        if not files:
            print(f"Warning: No files found in category '{category}'!")
            continue

        unique_files, dropped = dedupe_by_content(files)
        if dropped:
            print(f"[{category}] Dropped {len(dropped)} duplicate image(s) "
                  f"(same content, different filename) before splitting:")
            for d in dropped:
                print(f"    - {os.path.basename(d)}")

        unique_files = sorted(unique_files)  # deterministic order before shuffling
        rng.shuffle(unique_files)

        total = len(unique_files)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])

        split_files = {
            'train': unique_files[:train_end],
            'val': unique_files[train_end:val_end],
            'test': unique_files[val_end:],
        }

        for split_name, file_list in split_files.items():
            dest_dir = os.path.join(output_base_dir, split_name, category)
            for src in file_list:
                shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))

        summary[category] = {
            'total_unique': total,
            'dropped_duplicates': len(dropped),
            'train': len(split_files['train']),
            'val': len(split_files['val']),
            'test': len(split_files['test']),
        }

    print("\nSplit summary (content-deduplicated, seed={}):".format(seed))
    for category, stats in summary.items():
        print(f"  {category}: {stats}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_dataset_dir = os.path.join(base_dir, 'archive (1)', 'brain_tumor_dataset')
    output_base_dir = os.path.dirname(os.path.abspath(__file__))

    split_dataset(original_dataset_dir, output_base_dir, seed=args.seed)
    print("\nDataset successfully split into train/val/test folders (leak-free, deduplicated).")
