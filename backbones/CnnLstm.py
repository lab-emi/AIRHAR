# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978926&tag=1
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Any, cast, Dict, List, Optional, Union
from einops import rearrange

class CNNLSTM(nn.Module):

    def __init__(self, hidden_size,init_weights=True, channels=1,image_height=240, image_width=224,num_classes=6):
        super(CNNLSTM, self).__init__()
        self.init_weights = init_weights
        self.hidden_size = hidden_size
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes
        self.CNN = nn.Sequential(
            nn.Conv1d(self.channels*image_height, self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_size), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.hidden_size, 2*self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(2*self.hidden_size), nn.ReLU(),
            nn.Conv1d(2*self.hidden_size, 4*self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(4*self.hidden_size), nn.ReLU(),
        )

        self.LSTM = nn.LSTM(input_size=4*self.hidden_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=False)
        self.out = nn.Sequential(nn.Flatten(),
                                nn.Linear(int(self.image_width/2)*self.hidden_size, self.num_classes),
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
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        if self.channels == 1:
            x = x.unsqueeze(1)
        x = rearrange(x, 'b c h w -> b (c h) w')
        x = self.CNN(x)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x, (h_0, c_0))
        x = self.out(x)
        return x

