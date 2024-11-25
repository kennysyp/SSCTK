import torch
import torch.nn as nn
import torch.optim as optim

# 定义降采样卷积层
class DownsampleConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=1):
        super(DownsampleConv, self).__init__() 

        #长宽减少为原来的一半
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x1 = self.conv1(x[0])
        x2 = self.conv2(x[1])
        x3 = self.conv3(x[2])
        x4 = self.conv4(x[3])
        

        # 将特征展平并转换维度 (batch_size, channels, height, width) -> (batch_size, num_patches, embedding_dim)

        x1 = x1.view(x1.size(0),x1.size(1),-1).permute(0,2,1)
        x2 = x2.view(x2.size(0),x2.size(1),-1).permute(0,2,1)
        x3 = x3.view(x3.size(0),x3.size(1),-1).permute(0,2,1)
        x4 = x4.view(x4.size(0),x4.size(1),-1).permute(0,2,1)
        return x1,x2,x3,x4
