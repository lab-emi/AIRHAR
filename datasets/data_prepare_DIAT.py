import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import argparse

def prepare_diat_dataset(path):
    # Base path and class names
    base_path = path
    output_dir = "datasets/DIAT"
    os.makedirs(output_dir, exist_ok=True)

    classes = [
        'Army crawling',
        'Army jogging',
        'Army marching',
        'Boxing',
        'Jumping with holding a gun',
        'Stone pelting-Grenades throwing'
    ]
    
    # Lists to store train and test data for all classes
    X_train_all = []
    y_train_all = []  # Will store integer labels
    X_val_all = []
    y_val_all = []   # Will store integer labels
    X_test_all = []
    y_test_all = []   # Will store integer labels
    
    # Process each class
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        print(f"Processing class: {class_name}")
        
        # Lists to store data for current class
        class_images = []
        class_labels = []
        
        # Process all files in the class directory
        for filename in os.listdir(class_path):
            if filename.startswith('figure') and filename.endswith('.jpg'):
                image_path = os.path.join(class_path, filename)
                
                try:
                    # Read and resize image to 224x224
                    img = Image.open(image_path)
                    img = img.resize((224, 224))
                    img_array = np.array(img)
                    
                    # Normalize image data to [0, 1] first
                    img_array = img_array.astype('float32')
                    
                    # Reshape and normalize each sample independently
                    img_array = np.transpose(img_array, (2, 0, 1))
                    # Calculate mean and std for this sample
                    # sample_mean = np.mean(img_array)
                    # sample_std = np.std(img_array)
                    # # Normalize the sample
                    # img_array = (img_array - sample_mean) / (sample_std + 1e-8)
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
                    class_images.append(img_array)
                    class_labels.append(class_idx)
                except Exception as e:
                    print(f"Skipping {image_path}: {str(e)}")
        
        print(f"Processed {len(class_images)} valid images for class {class_name}")
        
        # Convert to numpy arrays
        X_class = np.array(class_images)
        y_class = np.array(class_labels)
        
        # Split data for this class into train+val and test first (80:20)
        X_trainval, X_test_class, y_trainval, y_test_class = train_test_split(
            X_class, y_class, test_size=0.2, shuffle=False
        )
        
        # Then split train+val into train and val (7:1 ratio from the 80%)
        X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(
            X_trainval, y_trainval, test_size=0.125, shuffle=False  # 0.125 of 80% = 10% of total
        )
        
        # Append to overall lists
        X_train_all.append(X_train_class)
        y_train_all.append(y_train_class)
        X_val_all.append(X_val_class)
        y_val_all.append(y_val_class)
        X_test_all.append(X_test_class)
        y_test_all.append(y_test_class)
    
    # Combine all classes
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val = np.concatenate(X_val_all, axis=0)
    y_val = np.concatenate(y_val_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)
    
    # Now convert combined labels to one-hot format
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_onehot = encoder.transform(y_val.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train_onehot.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val_onehot.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test_onehot.shape}")
    
    # Save to h5py file
    output_file = os.path.join(output_dir, 'diat_data.h5')
    with h5py.File(output_file, 'w') as f:
        # Create train group
        train_group = f.create_group('train')
        train_group.create_dataset('spectrograms', data=X_train)
        train_group.create_dataset('labels', data=y_train_onehot)
        
        # Create validation group
        val_group = f.create_group('val')
        val_group.create_dataset('spectrograms', data=X_val)
        val_group.create_dataset('labels', data=y_val_onehot)
        
        # Create test group
        test_group = f.create_group('test')
        test_group.create_dataset('spectrograms', data=X_test)
        test_group.create_dataset('labels', data=y_test_onehot)
    
    print(f"\nDataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the DIAT dataset.')
    parser.add_argument('--path', type=str, help='The base directory path for the dataset')
    args = parser.parse_args()

    prepare_diat_dataset(args.path)