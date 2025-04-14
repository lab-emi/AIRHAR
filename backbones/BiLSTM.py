# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9130759&tag=1
import torch
import torch.nn as nn

from einops import rearrange


class DopplerBiLSTM(nn.Module):

    def __init__(self, init_weights=True,hidden_size=500, channels=1, image_height=240, image_width=224,num_classes=6):
        super(DopplerBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes

        self.LSTM1 = nn.LSTM(input_size=channels*image_height,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=False)
        self.LSTM2 = nn.LSTM(input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=True)
        self.bn = nn.BatchNorm1d(channels*image_height)
        self.classifier = nn.Sequential(
            nn.Linear(2*image_width*self.hidden_size, num_classes),
            # nn.Softmax(dim=1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.channels == 1:
            x = x.unsqueeze(1)
        x = rearrange(x, 'b c h w -> b (c h) w')
        x = self.bn(x)
        x = x.transpose(2, 1)
        b, s, d = x.size()
        x, _ = self.LSTM1(x)
        x, _ = self.LSTM2(x)
        x = x.reshape(b, -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)





