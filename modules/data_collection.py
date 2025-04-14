import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler



class Radardataloader(Dataset):
    """
    Dataset class for Glasgow radar data.
    Loads spectrograms and their corresponding one-hot encoded labels from h5py file.
    """
    def __init__(self, root='Glasgow', subset='train'):
        """
        Args:
            root (str): Root directory containing the h5py file
            subset (str): Either 'train' or 'test'
        """
        super(Radardataloader, self).__init__()
        assert subset in ['train', 'test']
        
        # Store h5py file path instead of loading the file
        self.h5_path = f"datasets/{root}/{root.lower()}_data.h5"
        self.subset = subset
        
        # Open file temporarily to get dataset length
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[f'{subset}/spectrograms'])
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: torch.Tensor of shape (time_steps, freq_bins)
            label: torch.Tensor of shape (time_steps, num_classes)
        """
        # Open file, get item, and close file for each access
        with h5py.File(self.h5_path, 'r') as f:
            input_data = torch.FloatTensor(f[f'{self.subset}/spectrograms'][idx])
            label = torch.FloatTensor(f[f'{self.subset}/labels'][idx])
        
        return input_data, label
    
    def __len__(self):
        return self.length

def load_Radardataset(step, root, subset, batch_size, feature_type, num_classes, num_gpus=1):
    """
    Get dataloader with distributed sampling
    
    Args:
        step (str): Type of processing step
        root (str): Root directory containing the data
        subset (str): Either 'train' or 'test'
        batch_size (int): Batch size
        feature_type (str): Type of features
        num_classes (int): Number of classes
        num_gpus (int): Number of GPUs for distributed training
    """
    dataset = Radardataloader(
        step=step,
        root=root,
        subset=subset,
        feature_type=feature_type,
        classes=num_classes
    )
    kwargs = {"batch_size": batch_size}
    
    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=(subset == 'train'), **kwargs)
    
    return dataloader

def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param

class RadarFrameDataset(Dataset):
    """
    Dataset class for handling radar data frames.
    Creates frames from spectrograms with specified length and stride.
    """
    def __init__(self, spectrograms, frame_length, stride=1, subset='train', continuous=False):
        """
        Args:
            spectrograms: Input spectrograms
            frame_length (int): Length of each frame
            stride (int): Step size between frames (default=1)
        """
        self.frame_length = frame_length
        self.stride = stride
        self.subset = subset
        self.continuous = continuous
        
        # Load entire dataset at initialization
        with h5py.File(spectrograms.h5_path, 'r') as f:
            self.spectrograms = torch.FloatTensor(f[f'{subset}/spectrograms'][:])
            self.labels = torch.FloatTensor(f[f'{subset}/labels'][:])
            data_length = self.spectrograms.shape[-1]
            self.num_frames = (data_length - frame_length) // stride + 1
        
        # Pre-create frames if continuous mode is enabled
        if self.continuous:
            if self.subset == 'train':
                self.frames = self.create_frames(self.spectrograms, self.frame_length, self.stride)
                self.frame_labels = self.create_frames(self.labels, self.frame_length, self.stride)
                self.frame_labels = self.frame_labels[:, :, :, int(self.frame_length/2)+1]
            else:
                self.frames = self.create_segments(self.spectrograms, self.frame_length, self.stride)
                self.frame_labels = self.create_segments(self.labels, self.frame_length, self.stride)
                self.frame_labels = self.frame_labels[:, :, :, int(self.frame_length/2)+1]
            
            if self.frames.shape[1] == 1:
                self.frames = self.frames.squeeze(1)
                self.frame_labels = self.frame_labels.squeeze(1)

    def __getitem__(self, idx):
        """Returns a frame and its corresponding label."""
        if self.continuous:
            return self.frames[idx], self.frame_labels[idx]
        else:
            return self.spectrograms[idx], self.labels[idx]
    
    def __len__(self):
        if self.continuous:
            return len(self.frames)
        else:
            # For non-continuous mode, calculate total number of possible frames
            data_length = self.spectrograms.shape[-1]
            num_frames = (data_length - self.frame_length) // self.stride + 1
            return len(self.spectrograms) * num_frames
    @staticmethod
    def create_frames(data, frame_length, stride):
        # Handle both 3D and 4D input data
        if len(data.shape) == 3:  # (batch, height, time)
            data = data.unsqueeze(1)  # Add channel dimension: (batch, channel=1, height, time)
        
        batch_size = data.shape[0]
        data_length = data.shape[-1]
        num_frames = (data_length - frame_length) // stride + 1
        
        # Reshape to have combined batch and frames as the first dimension
        # New shape: (batch_size * num_frames, channel, height, frame_length)
        frames = torch.zeros((batch_size * num_frames, data.shape[1], data.shape[2], frame_length))
        
        for b in range(batch_size):
            for i in range(num_frames):
                frames[b * num_frames + i] = data[b, :, :, i * stride: i * stride + frame_length]

        return frames
    @staticmethod
    def create_segments(data, frame_length, stride):
        # Handle both 3D and 4D input data
        if len(data.shape) == 3:  # (batch, height, time)
            data = data.unsqueeze(1)  # Add channel dimension: (batch, channel=1, height, time)
        
        batch_size = data.shape[0]
        data_length = data.shape[-1]
        num_frames = (data_length - frame_length) // stride + 1
        
        # Reshape to have combined batch and frames as the first dimension
        # Shape: (batch_size * num_frames, channel, height, frame_length)
        frames = torch.zeros((batch_size * num_frames, data.shape[1], data.shape[2], frame_length))
        
        for b in range(batch_size):
            for j in range(num_frames):
                frames[b * num_frames + j] = data[b, :, :, j * stride: j * stride + frame_length]
        
        return frames

