import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from networks.FFC_3D import FFC3DConv  

class DoubleConv(nn.Module):  
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5):  
        super().__init__()  
        
        # 计算本地和全局通道数  
        out_cg = int(out_channels * ratio_gout)  
        out_cl = out_channels - out_cg  
        
        self.conv1 = FFC3DConv(in_channels, out_channels, kernel_size=3,   
                              ratio_gin=ratio_gin, ratio_gout=ratio_gout, padding=1)  
        # 分别为本地和全局特征创建BatchNorm  
        self.bn1_l = nn.BatchNorm3d(out_cl)  
        self.bn1_g = nn.BatchNorm3d(out_cg) if out_cg > 0 else None  
        self.relu1 = nn.ReLU(inplace=True)  
        
        self.conv2 = FFC3DConv(out_channels, out_channels, kernel_size=3,  
                              ratio_gin=ratio_gout, ratio_gout=ratio_gout, padding=1)  
        # 分别为本地和全局特征创建BatchNorm  
        self.bn2_l = nn.BatchNorm3d(out_cl)  
        self.bn2_g = nn.BatchNorm3d(out_cg) if out_cg > 0 else None  
        self.relu2 = nn.ReLU(inplace=True)  
        
        self.ratio_gout = ratio_gout  

    def forward(self, x):  
        # 第一个卷积块  
        x_l, x_g = self.conv1(x)  
        
        # 分别进行BatchNorm  
        if x_l is not None:  
            x_l = self.bn1_l(x_l)  
            x_l = self.relu1(x_l)  
        
        if x_g is not None and self.bn1_g is not None:  
            x_g = self.bn1_g(x_g)  
            x_g = self.relu1(x_g)  
            
        # 第二个卷积块  
        x = (x_l, x_g)  
        x_l, x_g = self.conv2(x)  
        
        # 分别进行BatchNorm  
        if x_l is not None:  
            x_l = self.bn2_l(x_l)  
            x_l = self.relu2(x_l)  
        
        if x_g is not None and self.bn2_g is not None:  
            x_g = self.bn2_g(x_g)  
            x_g = self.relu2(x_g)  
        
        # 如果需要返回单个张量（比如在模型最后一层）  
        if self.ratio_gout == 0:  
            return x_l  
        else:  
            return (x_l, x_g)  

class Down(nn.Module):  
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5):  
        super().__init__()  
        self.maxpool = nn.MaxPool3d(2)  
        self.conv = DoubleConv(in_channels, out_channels, ratio_gin=ratio_gin, ratio_gout=ratio_gout)  

    def forward(self, x):  
        if isinstance(x, tuple):  
            x_l, x_g = x  
            x_l = self.maxpool(x_l) if x_l is not None else None  
            x_g = self.maxpool(x_g) if x_g is not None else None  
            x = (x_l, x_g)  
        else:  
            x = self.maxpool(x)  
        return self.conv(x)  

class Up(nn.Module):  
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5, bilinear=True):  
        super().__init__()  

        if bilinear:  
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  
        else:  
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
            
        self.conv = DoubleConv(in_channels, out_channels, ratio_gin=ratio_gin, ratio_gout=ratio_gout)  

    def forward(self, x1, x2):  
        if isinstance(x1, tuple):  
            x1_l, x1_g = x1  
            x1_l = self.up(x1_l) if x1_l is not None else None  
            x1_g = self.up(x1_g) if x1_g is not None else None  
        else:  
            x1_l = self.up(x1)  
            x1_g = None  

        if isinstance(x2, tuple):  
            x2_l, x2_g = x2  
        else:  
            x2_l = x2  
            x2_g = None  

        # 处理padding  
        if x1_l is not None and x2_l is not None:  
            diffZ = x2_l.size()[2] - x1_l.size()[2]  
            diffY = x2_l.size()[3] - x1_l.size()[3]  
            diffX = x2_l.size()[4] - x1_l.size()[4]  
            x1_l = F.pad(x1_l, [diffX // 2, diffX - diffX // 2,  
                               diffY // 2, diffY - diffY // 2,  
                               diffZ // 2, diffZ - diffZ // 2])  
            
        if x1_g is not None and x2_g is not None:  
            diffZ = x2_g.size()[2] - x1_g.size()[2]  
            diffY = x2_g.size()[3] - x1_g.size()[3]  
            diffX = x2_g.size()[4] - x1_g.size()[4]  
            x1_g = F.pad(x1_g, [diffX // 2, diffX - diffX // 2,  
                               diffY // 2, diffY - diffY // 2,  
                               diffZ // 2, diffZ - diffZ // 2])  

        # 合并特征  
        if x2_l is not None:  
            x_l = torch.cat([x2_l, x1_l], dim=1) if x1_l is not None else x2_l  
        else:  
            x_l = x1_l  
            
        if x2_g is not None:  
            x_g = torch.cat([x2_g, x1_g], dim=1) if x1_g is not None else x2_g  
        else:  
            x_g = x1_g  

        x = (x_l, x_g)  
        return self.conv(x) 

class OutConv(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(OutConv, self).__init__()  
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)  

    def forward(self, x):  
        return self.conv(x)  

class FFCUNet3D(nn.Module):  
    def __init__(self, n_channels=1, n_classes=2, ratio=0.5, bilinear=True):  
        super(FFCUNet3D, self).__init__()  
        self.n_channels = n_channels  
        self.n_classes = n_classes  
        self.bilinear = bilinear  
        factor = 2 if bilinear else 1  

        self.inc = DoubleConv(n_channels, 64, ratio_gin=0, ratio_gout=ratio)  
        self.down1 = Down(64, 128, ratio_gin=ratio, ratio_gout=ratio)  
        self.down2 = Down(128, 256, ratio_gin=ratio, ratio_gout=ratio)  
        self.down3 = Down(256, 512, ratio_gin=ratio, ratio_gout=ratio)  
        self.down4 = Down(512, 1024 // factor, ratio_gin=ratio, ratio_gout=ratio)  
        
        self.up1 = Up(1024, 512 // factor, ratio_gin=ratio, ratio_gout=ratio, bilinear=bilinear)  
        self.up2 = Up(512, 256 // factor, ratio_gin=ratio, ratio_gout=ratio, bilinear=bilinear)  
        self.up3 = Up(256, 128 // factor, ratio_gin=ratio, ratio_gout=ratio, bilinear=bilinear)  
        self.up4 = Up(128, 64, ratio_gin=ratio, ratio_gout=0, bilinear=bilinear)  
        
        self.outc = OutConv(64, n_classes)  

    def forward(self, x):  
        x1 = self.inc(x)  
        x2 = self.down1(x1)  
        x3 = self.down2(x2)  
        x4 = self.down3(x3)  
        x5 = self.down4(x4)  
        
        x = self.up1(x5, x4)  
        x = self.up2(x, x3)  
        x = self.up3(x, x2)  
        x = self.up4(x, x1)  
        
        logits = self.outc(x)  
        return logits  

def get_ffc_unet_3d(n_channels=1, n_classes=2, ratio=0.5, bilinear=True):  
    return FFCUNet3D(n_channels, n_classes, ratio, bilinear)
