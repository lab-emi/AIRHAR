__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import json
import os
import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Any, Callable
from torch import optim
from torch.utils.data import DataLoader
from arguments import get_arguments
from modules.paths import create_folder, gen_log_stat, gen_dir_paths, gen_file_paths
from modules.train_funcs import net_train, net_eval,calculate_metrics
from modules.loggers import PandasLogger
from sklearn import metrics
import pandas as pd


class Project:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log
        self.log_all = {}
        self.log_train = {}
        self.log_val = {}
        self.log_test = {}

        # Load Hyperparameters
        self.args = get_arguments()
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

        # Load Specifications
        self.load_spec()

        # Hardware Info
        self.num_cpu_threads = os.cpu_count()

        # Configure Reproducibility
        self.reproducible()

        ###########################################################################################################
        #  Model ID, Paths of folders and log files and Logger
        ###########################################################################################################
        # Create Folders
        dir_paths = gen_dir_paths(self.args)
        self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best = dir_paths
        create_folder([self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best])


    def gen_classifier_model_id(self, n_net_params):
        m_dict = {'S': f"{self.seed}",
                   'M': self.Classification_backbone.upper(),
                   'B': f"{self.batch_size:d}",
                   'LR': f"{self.lr:.4f}",
                   'H': f"{self.Classification_hidden_size:d}",
                   'P': f"{n_net_params:d}",
                   'FL': f"{self.frame_length:d}",
                   'ST': f"{self.stride:d}",
                   }
        dict_model_id = dict(list(m_dict.items()))


        list_model_id = []
        for item in list(dict_model_id.items()):
            list_model_id += list(item)
        model_id = '_'.join(list_model_id)
        model_id = 'CL_' + model_id
        return model_id

    def build_logger(self, model_id: str):
        # Get Save and Log Paths
        file_paths = gen_file_paths(self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best, model_id)
        self.path_save_file_best, self.path_log_file_hist, self.path_log_file_best = file_paths
        print("::: Best Model Save Path: ", self.path_save_file_best)
        print("::: Log-History     Path: ", self.path_log_file_hist)
        print("::: Log-Best        Path: ", self.path_log_file_best)

        # Instantiate Logger for Recording Training Statistics
        self.logger = PandasLogger(path_save_file_best=self.path_save_file_best,
                                   path_log_file_best=self.path_log_file_best,
                                   path_log_file_hist=self.path_log_file_hist,
                                   precision=self.log_precision)

    def reproducible(self):
        rnd.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.autograd.set_detect_anomaly(True)

        if self.re_level == 'soft':
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # re_level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print("::: Are Deterministic Algorithms Enabled: ", torch.are_deterministic_algorithms_enabled())
        print("--------------------------------------------------------------------")

    def load_spec(self):
        # Get relative path to the spec file
        path_spec = os.path.join('datasets', self.dataset_name, 'spec.json')
    
        # Load the spec
        with open(path_spec) as config_file:
            spec = json.load(config_file)
        for k, v in spec.items():
            setattr(self, k, v)
            self.hparams[k] = v

    def add_arg(self, key: str, value: Any):
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def set_device(self):
        # Find Available GPUs
        if self.accelerator == 'cuda' and torch.cuda.is_available():
            idx_gpu = self.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print("--------------------------------------------------------------------")
        elif self.accelerator == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.accelerator == 'cpu':
            device = torch.device("cpu")
            print("::: Available GPUs: None")
            print("--------------------------------------------------------------------")
        else:
            raise ValueError(f"The select device {self.accelerator} is not supported.")
        self.add_arg("device", device)
        return device

    def build_dataloaders(self):
        # Load Dataset
        from modules.data_collection import Radardataloader, RadarFrameDataset
        data_path = os.path.join(self.dataset_name)
        
        # Create datasets with lazy loading
        train_dataset = Radardataloader(
            root=data_path,
            subset='train'
        )
        
        test_dataset = Radardataloader(
            root=data_path,
            subset='test'
        )
        
        # Create frame datasets - only transform the data when needed
        train_frame_dataset = RadarFrameDataset(
            spectrograms=train_dataset,  # Pass the dataset object instead of loaded data
            frame_length=self.frame_length,
            stride=self.stride,
            subset='train',
            continuous=self.continuous
        )

        test_frame_dataset = RadarFrameDataset(
            spectrograms=test_dataset,  # Pass the dataset object instead of loaded data
            frame_length=self.frame_length,
            stride=self.stride,
            subset='test',
            continuous=self.continuous
        )
        
        # Use num_workers for parallel data loading
        train_loader = torch.utils.data.DataLoader(
            train_frame_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=min(4, os.cpu_count()),  # Use multiple workers for parallel loading
            pin_memory=True  # Pin memory for faster data transfer to GPU
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_frame_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            # num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        return train_loader, test_loader


    def build_criterion(self):
        if self.loss_type == 'WeightedCrossEntropy':
            # Calculate class weights based on inverse frequency
            # You'll need to compute these weights from your training data
            # Example weights - adjust these based on your actual class distribution
            class_weights = torch.tensor([
                1.15,  # Adjust these weights based on your confusion matrix
                1.12,  # You can use inverse frequency or sqrt/log of inverse frequency
                1.0,  # to balance the classes
                1.1,
                1.15,
                1.08,
            ]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            dict_loss = {'l2': nn.MSELoss(),
                         'l1': nn.L1Loss(),
                         'CrossEntropy': nn.CrossEntropyLoss()
                         }
            loss_func_name = self.loss_type
            try:
                criterion = dict_loss[loss_func_name]
            except KeyError:
                raise AttributeError('Please use a valid loss function. Check arguments.py.')
        
        self.add_arg("criterion", criterion)
        return criterion

    def build_optimizer(self, net: nn.Module):

        # Optimizer
        if self.opt_type == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=self.lr)
        elif self.opt_type == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt_type == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), lr=self.lr)
        elif self.opt_type == 'adamw':
            optimizer = optim.AdamW(net.parameters(), lr=self.lr)
        elif self.opt_type == 'adabound':
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)
            optimizer = adabound.AdaBound(net.parameters(), lr=self.lr, final_lr=0.1)
        else:
            raise RuntimeError('Please use a valid optimizer.')

        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=self.decay_factor,
                                                            patience=self.patience,
                                                            verbose=True,
                                                            threshold=1e-4,
                                                            min_lr=self.lr_end)
        return optimizer, lr_scheduler

    def train(self, net: nn.Module, criterion: Callable, optimizer: optim.Optimizer, lr_scheduler,
              train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, best_model_metric: str) -> None:
        if self.step == 'classify':
            best_model_metric = 'accuracy'
        # Timer
        start_time = time.time()
        accuracy = []
        # Track best model predictions and ground truth
        best_predictions = None
        best_ground_truth = None
        best_metric_value = 0
        # Epoch loop
        print("Starting training...")
        for epoch in range(self.n_epochs):
            # -----------
            # Train
            # -----------
            net = net_train(log=self.log_train,
                            net=net,
                            optimizer=optimizer,
                            criterion=criterion,
                            dataloader=train_loader,
                            grad_clip_val=self.grad_clip_val,
                            device=self.device)

            # -----------
            # Validation
            # -----------
            if self.eval_val:
                _, prediction, ground_truth = net_eval(log=self.log_val,
                                                       net=net,
                                                       criterion=criterion,
                                                       dataloader=val_loader,
                                                       device=self.device)
                self.log_val = calculate_metrics(best_model_metric, self.log_val, prediction, ground_truth, self.num_classes, self.continuous)

            # -----------
            # Test
            # -----------
            if self.eval_test:
                _, prediction, ground_truth = net_eval(log=self.log_test,
                                                       net=net,
                                                       criterion=criterion,
                                                       dataloader=test_loader,
                                                       device=self.device)
                self.log_test = calculate_metrics(best_model_metric, self.log_test, prediction, ground_truth, self.num_classes, self.continuous)

                # Update best predictions if current model is the best
                if self.log_test[best_model_metric] > best_metric_value:
                    best_metric_value = self.log_test[best_model_metric]
                    best_predictions = prediction
                    best_ground_truth = ground_truth

            ###########################################################################################################
            # Logging & Saving
            ###########################################################################################################

            # Generate Log Dict
            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60.0
            self.log_all = gen_log_stat(self.args, elapsed_time_minutes, net, optimizer, epoch, self.log_train,
                                        self.log_val, self.log_test)

            # Write Log
            self.logger.write_log(self.log_all)

            # Save best model
            # best_net = net.dpd_model if self.step == 'classification' else net
            best_net = net
            self.logger.save_best_model(net=best_net, epoch=epoch, val_stat=self.log_test, metric_name=best_model_metric)
            if self.step == 'classify':
                accuracy.append(self.log_test[best_model_metric])
            ###########################################################################################################
            # Learning Rate Schedule
            ###########################################################################################################
            # Schedule at the beginning of retrain
            lr_scheduler_criteria = self.log_test[best_model_metric]
            if self.lr_schedule:
                lr_scheduler.step(lr_scheduler_criteria)
        print("Training Completed...")
        print(" ")