# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978926&tag=1
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Any, cast, Dict, List, Optional, Union
from einops import rearrange

class CNNGRU(nn.Module):

    def __init__(self, hidden_size,init_weights=True, channels=1,image_height=240, image_width=224,num_classes=6):
        super(CNNGRU, self).__init__()
        self.init_weights = init_weights
        self.hidden_size = hidden_size
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes
        self.CNN = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=(10,5), padding=(5,2)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding="same"),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.GRU1 = nn.GRU(input_size=int(self.image_height/2),
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.GRU2 = nn.GRU(input_size=hidden_size*2,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.out = nn.Sequential(nn.Flatten(),
                                nn.Linear(2*self.image_width*self.hidden_size, self.num_classes),
                                )
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        if self.channels == 1:
            x = x.unsqueeze(1)
        x = self.CNN(x)
        x = nn.functional.max_pool2d(x, kernel_size=(1, 1))
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.GRU1(x, h_0)
        x, _ = self.GRU2(x, h_0)
        x = self.out(x)
        return x



   