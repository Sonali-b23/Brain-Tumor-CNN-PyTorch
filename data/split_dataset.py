import os
import shutil
import random

# Paths
original_dataset_dir = r'/mnt/c/Users/sonal/OneDrive/Desktop/ComputerVsionProject/Brain-Tumor-Detection/archive (1)/brain_tumor_dataset'

output_dirs = ['train', 'val', 'test']
categories = ['yes', 'no']

# Split ratios
split_ratios = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}

# Create output directories if they don't exist
for split in output_dirs:
    for category in categories:
        os.makedirs(os.path.join(split, category), exist_ok=True)

# Loop over categories and split files
for category in categories:
    category_path = os.path.join(original_dataset_dir, category)
    
    # Check if category folder exists
    if not os.path.exists(category_path):
        print(f"Warning: Category folder '{category}' not found!")
        continue  # Skip to next category if this one is missing

    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    
    if not files:
        print(f"Warning: No files found in category '{category}'!")
        continue  # Skip if no files are found in this category

    random.shuffle(files)  # Shuffle to randomize the split
    
    total = len(files)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])
    
    # Define file splits
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    # Copy files function
    def copy_files(file_list, split_name):
        for file in file_list:
            src = os.path.join(category_path, file)
            dst = os.path.join(split_name, category, file)
            shutil.copy2(src, dst)
    
    # Copy to respective folders
    print(f"Processing category: {category}...")
    print(f"  Train set: {len(train_files)} files")
    print(f"  Validation set: {len(val_files)} files")
    print(f"  Test set: {len(test_files)} files")
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

print("Dataset successfully split into train/val/test folders!")
