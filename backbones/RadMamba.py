#"""VisionMambaBlock module."""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from backbones.SSM import SSM
from einops.layers.torch import Reduce
import matplotlib
matplotlib.use('Agg')  # Set to non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import signal

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)
# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int, num_classes: int):
    # """
    # Creates a head for the output layer of a model.

    # Args:
    #     dim (int): The input dimension of the head.
    #     num_classes (int): The number of output classes.

    # Returns:
    #     nn.Sequential: The output head module.
    # """
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class VisionEncoderMambaBlock(nn.Module):
    # """
    # VisionMambaBlock is a module that implements the Mamba block from the paper
    # Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    # State Space Model

    # Args:
    #     dim (int): The input dimension of the input tensor.
    #     dt_rank (int): The rank of the state space model.
    #     dim_inner (int): The dimension of the inner layer of the
    #         multi-head attention.
    #     d_state (int): The dimension of the state space model.


    # Example:
    # >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
    #         dim_inner=512, d_state=256)
    # >>> x = torch.randn(1, 32, 256)
    # >>> out = block(x)
    # >>> out.shape
    # torch.Size([1, 32, 256])
    # """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        
        
        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1,
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm_forward = nn.LayerNorm(dim)
        self.norm_backward = nn.LayerNorm(dim)
        


        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm1 = SSM(dim, dt_rank, dim_inner, d_state)
        self.ssm2 = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        # Linear layer for x
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        self.proj3 = nn.Conv1d(dim, dim, kernel_size=1)


        # Softplus
        self.softplus = nn.Softplus()
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)
        
        # Visualize input after normalization

        x = rearrange(x, "b s d -> b d s")
        z1 = self.proj1(x)
        x = self.proj2(x)
        z1 = rearrange(z1, "b d s -> b s d")

        

        # Matmul
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.norm_forward,
            self.ssm1,
        )

        # backward conv1d
        x2 = self.process_direction(
            torch.flip(x,dims=[2]),
            self.backward_conv1d,
            self.norm_backward,
            self.ssm2,
        )
        
        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z
        


        x3 = rearrange(x1+x2, "b s d -> b d s")
        out = self.proj3(x3)
        out = rearrange(out, "b d s -> b s d")
                
        # Residual connection
        return out + skip

    def process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        norm:nn.LayerNorm,
        ssm: SSM,
    ):
        x = conv1d(x)
        # print(f"Conv1d: {x}")
        x = rearrange(x, "b d s -> b s d")
        x = norm(x)
        x = ssm(x,pscan=True)

        return x

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'conv' in name:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'bn' in name:
                if 'weight' in name:
                    nn.init.constant_(param, 1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'fc' in name or 'linear' in name:
                if 'weight' in name:
                    nn.init.normal_(param, 0, 0.01)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'ln' in name or 'layernorm' in name:
                if 'weight' in name:
                    nn.init.constant_(param, 1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)


class RadMamba(nn.Module):
    # """
    # RadMamba model implementation.

    # Args:
    #     dim (int): Dimension of the model.
    #     dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
    #     dim_inner (int, optional): Inner dimension of the model. Defaults to None.
    #     d_state (int, optional): State dimension of the model. Defaults to Nosne.
    #     num_classes (int, optional): Number of output classes. Defaults to None.
    #     image_size (int, optional): Size of the input image. Defaults to 224.
    #     patch_size (int, optional): Size of the image patch. Defaults to 16.
    #     channels (int, optional): Number of image channels. Defaults to 3.
    #     dropout (float, optional): Dropout rate. Defaults to 0.1.
    #     depth (int, optional): Number of encoder layers. Defaults to 12.

    # Attributes:
    #     dim (int): Dimension of the model.
    #     dt_rank (int): Rank of the dynamic tensor.
    #     dim_inner (int): Inner dimension of the model.
    #     d_state (int): State dimension of the model.
    #     num_classes (int): Number of output classes.
    #     image_size (int): Size of the input image.
    #     patch_size (int): Size of the image patch.
    #     channels (int): Number of image channels.
    #     dropout (float): Dropout rate.
    #     depth (int): Number of encoder layers.
    #     to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
    #     dropout (nn.Dropout): Dropout module.
    #     cls_token (nn.Parameter): Class token parameter.
    #     to_latent (nn.Identity): Identity module for latent representation.
    #     layers (nn.ModuleList): List of encoder layers.
    #     output_head (output_head): Output head module.

    # """

    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_height: int = 240,
        image_width: int = 224,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        channel_confusion_layer: int = 1,
        channel_confusion_out_channels: int = 3,
        time_downsample_factor: int = 4,
        optional_avg_pool: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.dropout = dropout
        self.depth = depth
        self.cd_out_channels = channel_confusion_out_channels
        self.cd_layer = channel_confusion_layer
        self.td_factor = time_downsample_factor
        self.optional_avg_pool = optional_avg_pool
        self.batch_norm = nn.BatchNorm2d(channels)
        if channel_confusion_layer == 2:
            self.CNN = nn.Sequential(nn.Conv2d(channels, self.cd_out_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(self.cd_out_channels),
                                     nn.Conv2d(self.cd_out_channels, self.cd_out_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(self.cd_out_channels),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.MaxPool2d(kernel_size=(1,self.td_factor), stride=(1,self.td_factor)),
        )
            h_factor = 2
            w_factor = self.td_factor*2
            image_height = int((image_height)/h_factor)  # Updated due to vertical pooling
            image_width = int((image_width)/w_factor)
            channels = self.cd_out_channels
        elif channel_confusion_layer == 1:
            self.CNN = nn.Sequential(nn.Conv2d(channels, self.cd_out_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(self.cd_out_channels),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.MaxPool2d(kernel_size=(1,self.td_factor), stride=(1,self.td_factor)))
            w_factor = self.td_factor*2
            if self.optional_avg_pool:
                self.CNN.append(nn.AvgPool2d(kernel_size=(1,self.td_factor), stride=(1,self.td_factor)))
                w_factor = self.td_factor*2*self.td_factor
            h_factor = 2
            image_height = int((image_height)/h_factor)  # Updated due to vertical pooling
            image_width = int((image_width)/w_factor)
            channels = self.cd_out_channels         
        
        patch_height = image_height
        patch_width = 1

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #
        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers with threshold
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        self.output_head = output_head(dim, num_classes)



    def forward(self, x: Tensor):
        if self.channels == 1:
            x = x.unsqueeze(1)
        x = self.batch_norm(x)
        x = self.CNN(x)
        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(x.device, dtype=x.dtype)
        x = self.dropout(x)
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
        x = self.to_latent(x)
        x = self.output_head(x)
        return x


