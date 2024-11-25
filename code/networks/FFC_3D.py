import torch  
import torch.nn as nn  

class FFCSE_block(nn.Module):  
    def __init__(self, channels, ratio=16):  
        super(FFCSE_block, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  
        self.fc1 = nn.Conv3d(channels, channels // ratio, kernel_size=1, padding=0)  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Conv3d(channels // ratio, channels, kernel_size=1, padding=0)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):  
        module_input = x  
        x = self.avg_pool(x)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        x = self.sigmoid(x)  
        return module_input * x  

class FourierUnit(nn.Module):  
    def __init__(self, in_channels, out_channels, groups=1):  
        super(FourierUnit, self).__init__()  
        self.groups = groups  
        self.conv_layer = torch.nn.Conv3d(in_channels=in_channels * 2,  
                                        out_channels=out_channels * 2,  
                                        kernel_size=1,  
                                        stride=1,  
                                        padding=0,  
                                        groups=self.groups,  
                                        bias=False)  
        self.bn = torch.nn.BatchNorm3d(out_channels * 2)  
        self.relu = torch.nn.ReLU(inplace=True)  

    def forward(self, x):  
        batch = x.shape[0]  

        # 3D FFT  
        ffted = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')  
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  
        
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  

        ffted = self.conv_layer(ffted)  
        ffted = self.relu(self.bn(ffted))  

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 5, 2).contiguous()  
        real = ffted[..., 0]  
        imag = ffted[..., 1]  

        complex_output = torch.complex(real, imag)  
        output = torch.fft.irfftn(complex_output, dim=(-3, -2, -1), norm='ortho')  

        return output  

class SpectralTransform(nn.Module):  
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):  
        super().__init__()  
        self.enable_lfu = enable_lfu  
        if stride == 2:  
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)  
        else:  
            self.downsample = nn.Identity()  

        self.stride = stride  
        self.conv1 = nn.Sequential(  
            nn.Conv3d(in_channels, out_channels //  
                     2, kernel_size=1, groups=groups, bias=False),  
            nn.BatchNorm3d(out_channels // 2),  
            nn.ReLU(inplace=True)  
        )  
        self.fu = nn.Sequential(  
            nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=1,  
                     groups=groups, bias=False),  
            nn.BatchNorm3d(out_channels // 2),  
            nn.ReLU(inplace=True)  
        )  
        if self.enable_lfu:  
            self.lfu = nn.Sequential(  
                nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=1,  
                         groups=groups, bias=False),  
                nn.BatchNorm3d(out_channels // 2),  
                nn.ReLU(inplace=True)  
            )  
        self.conv2 = nn.Conv3d(  
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)  

    def forward(self, x):  
        try:  
            # 降采样  
            x = self.downsample(x)  
            
            # 减少内存使用  
            x = x.contiguous()  
            batch = x.shape[0]  

            # 使用torch.cuda.empty_cache()清理GPU内存  
            if x.is_cuda:  
                torch.cuda.empty_cache()  

            # 3D FFT  
            ffted = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')  
            
            # 分离实部和虚部  
            real = ffted.real  
            imag = ffted.imag  
            ffted = torch.stack((real, imag), dim=-1)  
            
            # 清理临时变量  
            del real, imag  
            if x.is_cuda:  
                torch.cuda.empty_cache()  

            ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  
            ffted_size = ffted.size()  
            
            # 减少维度处理内存  
            ffted = ffted.view(batch, -1, ffted_size[3],  
                             ffted_size[4], ffted_size[5])  

            ffted = self.conv1(ffted)  
            ffted = self.fu(ffted)  

            if self.enable_lfu:  
                ffted = self.lfu(ffted)  

            ffted = self.conv2(ffted)  

            # 恢复形状  
            ffted = ffted.view(batch, -1, 2, ffted_size[3],  
                             ffted_size[4], ffted_size[5])  
            ffted = ffted.permute(0, 1, 3, 4, 5, 2).contiguous()  

            # IFFT  
            ifft_shape_slice = x.shape[-3:]  
            output = torch.fft.irfftn(  
                torch.complex(ffted[..., 0], ffted[..., 1]),  
                s=ifft_shape_slice,  
                dim=(-3, -2, -1),  
                norm='ortho'  
            )  

            return output  

        except RuntimeError as e:  
            print(f"Error in SpectralTransform: {e}")  
            # 如果FFT失败，返回输入（降级处理）  
            return self.downsample(x) 

class FFC3DConv(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size,  
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=1,  
                 dilation=1, groups=1, bias=False, enable_lfu=True):  
        super(FFC3DConv, self).__init__()  

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."  
        self.stride = stride  

        in_cg = int(in_channels * ratio_gin)  
        in_cl = in_channels - in_cg  
        out_cg = int(out_channels * ratio_gout)  
        out_cl = out_channels - out_cg  

        self.ratio_gin = ratio_gin  
        self.ratio_gout = ratio_gout  
        self.global_in_num = in_cg  

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv3d  
        self.convl2l = module(in_cl, out_cl, kernel_size,  
                            stride, padding, dilation, groups, bias)  
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv3d  
        self.convl2g = module(in_cl, out_cg, kernel_size,  
                            stride, padding, dilation, groups, bias)  
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv3d  
        self.convg2l = module(in_cg, out_cl, kernel_size,  
                            stride, padding, dilation, groups, bias)  
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform  
        self.convg2g = module(in_cg, out_cg, stride, 1, enable_lfu)  

    def forward(self, x):  
        x_l, x_g = x if type(x) is tuple else (x, 0)  
        out_xl, out_xg = 0, 0  

        if self.ratio_gout != 1:  
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)  
        if self.ratio_gout != 0:  
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)  

        return out_xl, out_xg
