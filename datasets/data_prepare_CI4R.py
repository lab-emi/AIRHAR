import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import List
import cv2
from scipy.fft import fft, fftshift
import argparse


def bin2np(filepath):
    """
    Convert the bin file to numpy dataset

    Here we choose the fourth Lanes
    """
    NTS = 256  # Number of time samples per sweep
    N = 32000
    with open(filepath) as file:
            data = np.fromfile(file, dtype=np.int16)
            data = data.reshape((8,-1), order='F')
            data = data[3, :] + 1j * data[7, :]  # The fourth Lanes
            data = data.reshape((NTS, -1), order='F')
    return data


def RGB_mD(data, window=256, nfft=4096, overlap=200):
    NTS = 256
    SweepTime = 40e-3  # Time for 1 frame=sweep
    NPpF = 128  # Number of pulses per frame
    NoF = 500  # Number of frames
    dT = SweepTime / NPpF
    prf = 1 / dT
    # Number of time samples per sweep
    # rangeFFT
    fftRawData = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    rp = fftRawData[NTS // 2:, :]
    # MTI filter

    # STFT
    rangedata=np.sum(rp[23:31, :], axis=0)
    shift = window-overlap
    N = (len(rangedata) - window - 1) // shift
    rp = np.zeros((nfft, N), dtype=complex)
    for i in range(N):
        start_idx = i * shift
        end_idx = start_idx + window
        tmp = np.fft.fft(np.multiply(rangedata[start_idx:end_idx], np.hanning(window)), nfft)
        rp[:, i] = tmp

    timeAxis = np.arange(1, NPpF * NoF + 1) * dT  # Time
    sx2 = np.abs(np.flipud(fftshift(rp, axes=0)))
    
    # Calculate doppSignMTI
    doppSignMTI = 20 * np.log10(np.abs(sx2 / np.max(sx2)))
    doppSignMTI = np.clip(doppSignMTI, -45, 0)
    # Resize to 224x224 using cv2
    doppSignMTI = cv2.resize(doppSignMTI, (224, 224), interpolation=cv2.INTER_CUBIC)

    return doppSignMTI

def get_identifiers(base_dir: str) -> List[str]:
    return list(set(name.split('_')[0] for name in os.listdir(base_dir)))


def split_files(files: List[str], train_ratio: float) -> tuple:
    total = len(files)
    train_num = int(train_ratio * total)
    test_num = (total - train_num)
    return files[:train_num], files[train_num:train_num+test_num]

def one_hot_encode(index: int, num_classes: int) -> np.ndarray:
    """
    Create a one-hot encoded vector.
    
    Args:
        index: The index to set to 1
        num_classes: Total number of classes
    
    Returns:
        One-hot encoded numpy array
    """
    encoding = np.zeros(num_classes)
    encoding[index] = 1
    return encoding

def prepare_alabma_dataset(path, train_ratio=0.8):
    base_dir = path
    identifiers = get_identifiers(base_dir)
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for idx, identifier in enumerate(identifiers):
        features = []
        labels = []
        dir_name = next((d for d in os.listdir(base_dir) 
                        if identifier in d and os.path.isdir(os.path.join(base_dir, d))), None)
        if not dir_name:
            continue

        dir_path = os.path.join(base_dir, dir_name)
        # Collect all files for this identifier
        for f in os.listdir(dir_path):
            if f.endswith('.bin'):
                file_path = os.path.join(dir_path, f)
                # Skip empty files
                if os.path.getsize(file_path) == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue
                data = bin2np(file_path)
                feature = RGB_mD(data)
                features.append(feature)
                labels.append(one_hot_encode(idx, len(identifiers)))

        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Split data for this identifier
        train_size = int(len(features)-12)
        X_train_class = features[:train_size]
        X_test_class = features[train_size:]
        y_train_class = labels[:train_size]
        y_test_class = labels[train_size:]

        X_train_list.append(X_train_class)
        X_test_list.append(X_test_class)
        y_train_list.append(y_train_class)
        y_test_list.append(y_test_class)

    # Concatenate all splits
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    X_train = np.array([
        (spec - spec.mean()) / (spec.std())
        for spec in X_train
    ])
    X_test = np.array([
        (spec - spec.mean()) / (spec.std())
        for spec in X_test
    ])

    # Save to h5py format
    output_dir = "datasets/CI4R"
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(os.path.join(output_dir, 'ci4r_data.h5'), 'w') as f:
        # Create train group
        train_group = f.create_group('train')
        train_group.create_dataset('spectrograms', data=X_train, dtype='float32')
        train_group.create_dataset('labels', data=y_train, dtype='float32')
        
        # Create test group
        test_group = f.create_group('test')
        test_group.create_dataset('spectrograms', data=X_test, dtype='float32')
        test_group.create_dataset('labels', data=y_test, dtype='float32')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Prepare the Alabma dataset.')
    # parser.add_argument('--path', type=str, help='The base directory path for the dataset')
    # args = parser.parse_args()
    path = "/home/yizhuo/PythonProject/AIHAR/data/CI4R7"
    prepare_alabma_dataset(path)
