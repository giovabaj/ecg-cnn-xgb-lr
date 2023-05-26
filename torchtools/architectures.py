from torch import nn


class GoodfellowBlock(nn.Module):
    """Single convolution block of the network"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, p_dropout, padding="same", maxpooling=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation_rate)
        self.batch_norm = nn.BatchNorm1d(out_channels,  momentum=0.1)
        self.maxpooling = maxpooling
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU()(x)
        if self.maxpooling:
            x = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)(x)
        x = self.dropout(x)
        return x


class GoodfellowNet(nn.Module):
    """Network module definition.
    """
    def __init__(self, block=GoodfellowBlock, len_input=5000, out_nodes=2, p_dropout=0.3):
        super().__init__()
        self.p_dropout = p_dropout
        in_channels = [12, 320, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 64]   # input channels of each block
        out_channels = in_channels[1:] + [64]                                           # output channels
        kernel_size = [24, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8]
        dilation_rate = [1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8]
        max_pooling = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        # Convolutional layers
        blocks = [block(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel_size[i],
                        dilation_rate=dilation_rate[i], maxpooling=max_pooling[i], p_dropout=self.p_dropout)
                  for i in range(13)]
        self.conv_blocks = nn.Sequential(*blocks)
        # Layers after convolutions
        self.avg_pooling = nn.AvgPool1d(kernel_size=int(len_input/8))
        self.batch_norm = nn.BatchNorm1d(64, momentum=0.1)
        self.linear = nn.Linear(in_features=64, out_features=out_nodes)

    def forward(self, ecg):
        out = self.conv_blocks(ecg)             # Convolutions
        out = self.avg_pooling(out).squeeze(2)  # Global average pooling
        out = self.batch_norm(out)              # Batch normalization
        out = self.linear(out)                  # Linear layer
        return out
