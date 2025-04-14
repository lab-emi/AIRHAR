import os
import shutil
import numpy as np
import h5py
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import cv2
import matplotlib.pyplot as plt
import argparse
def prepare_glasgow_dataset(path):
    # Create output directory if it doesn't exist
    output_dir = "datasets/UoG20"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load .mat files
    specs = loadmat(path + '/Spectrograms.mat')['Doppler2_MM']  # Assuming 'spectrograms' is the variable name
    labels = loadmat(path + '/Label.mat')['UpdatedLabel']  # Assuming 'Labels' is the variable name
    
    # Reshape data to combine all participants
    # specs shape: (16, 3, time) -> (48, time)
    # labels shape: (16, 3, time) -> (48, time)
    X = np.array([spec for participant_specs in specs for spec in participant_specs])
    y = np.array([label for participant_labels in labels for label in participant_labels])
    

    X =X[:,:240,:]
        # # Normalize X: scale to [0, 1] range for each sample independently
    X = np.array([
        (spec - spec.mean()) / (spec.std())
        for spec in X
    ])
    # Save spectrograms as images and reload them with specific dimensions
    temp_dir = os.path.join(output_dir, 'temp_images')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Visualize first spectrogram before and after resizing
    # Plot original spectrogram
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(X[16], aspect='auto', cmap='viridis')
    plt.title(f'Original Spectrogram\nShape: {X[16].shape}')
    plt.colorbar()
    plt.savefig(os.path.join(temp_dir, 'original_spectrogram.png'))
    plt.close()
    
    # Process each time step separately for one-hot encoding
    time_steps = y.shape[2]  # Get number of time steps
    y_reshaped = y.reshape(-1, 1)  # Flatten to (48*time_steps, 1)
    
    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.arange(1, 7).reshape(-1, 1))
    y_onehot = encoder.transform(y_reshaped)
    
    # Reshape back to include time dimension (48, time_steps, 6)
    y_onehot = y_onehot.reshape(-1, time_steps, 6)
    y_onehot = y_onehot.transpose(0, 2, 1)
    
    # Split into training (87.5%) and test (12.5%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=2/16, random_state=42, shuffle=False
    )
    

    
    # Save to h5py format
    with h5py.File(os.path.join(output_dir, 'uog20_data.h5'), 'w') as f:
        # Create train group
        train_group = f.create_group('train')
        train_group.create_dataset('spectrograms', data=X_train)
        train_group.create_dataset('labels', data=y_train)
        
        # Create test group
        test_group = f.create_group('test')
        test_group.create_dataset('spectrograms', data=X_test)
        test_group.create_dataset('labels', data=y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the Glasgow dataset.')
    parser.add_argument('--path', type=str, help='The base directory path for the dataset')
    args = parser.parse_args()

    prepare_glasgow_dataset(args.path)


