import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

class TimeceptionBlock(nn.Module):
    def __init__(self, channels, groups=4, squeeze=4):
        super(TimeceptionBlock, self).__init__()
        self.groups = groups
        self.squeeze = squeeze
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # self.branch1_1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1), padding=(1, 0), ceil_mode=False)
        self.branch1_1 = nn.Conv2d(channels // groups, channels // groups, kernel_size=(1, 1), stride=1, padding=(0, 0),
                                   groups=channels // groups)
        self.branch2_1 = nn.Conv2d(channels // groups, channels // groups, kernel_size=(3, 1), stride=1, padding=(1, 0),
                                   groups=channels // groups)
        self.branch3_1 = nn.Conv2d(channels // groups, channels // groups, kernel_size=(5, 1), stride=1, padding=(2, 0),
                                   groups=channels // groups)
        self.branch4_1 = nn.Conv2d(channels // groups, channels // groups, kernel_size=(7, 1), stride=1, padding=(3, 0),
                                   groups=channels // groups)
        self.conv = nn.Conv2d(channels // groups, channels // groups // squeeze, kernel_size=1, stride=1, bias=False)
        self.branch5 = self.conv
        self.maxpool_s2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv_all = nn.Conv2d(channels*5//squeeze, channels, kernel_size=1, stride=1,bias=False)

    def forward(self, x):
        residual = x
        B, C, H, W = x.size()
        T = 8
        B = B // T
        x = x.view(-1, T, C, H, W)  # ;print('x.size() ',x.size())
        assert C % self.groups == 0
        stripe_c = int(C / self.groups)
        feat_list = []
        for i in range(self.groups):
            feat0 = x[:, :, i * stripe_c:(i + 1) * stripe_c, :, :]  # [4, 8, 512, 7, 7]  512
            sub_C = C // self.groups
            feat = feat0.contiguous().view(B, T, sub_C, H * W).permute(0, 3, 2, 1)  # ;print('feat ',feat.size()) #[4, 49, 512, 8]
            feat = feat.contiguous().view(B * H * W, sub_C, T, 1)

            out1_b = self.branch1_1(feat)  # ;print(i,'branch1: ',out1.size())
            out1 = out1_b.contiguous().view(B, H, W, sub_C, T, 1).permute(0, 4, 5, 3, 1, 2)
            out1 = out1.contiguous().view(B * T, sub_C, H, W)
            out1 = self.conv(out1)  # ;print(i,'out1: ',out1.size())

            out2_b = self.branch2_1(feat)  # ;print(i,'branch2: ',out2.size()) # [196, 512, 8, 1]
            out2 = (out2_b + out1_b).contiguous().view(B, H, W, sub_C, T, 1).permute(0, 4, 5, 3, 1, 2)  # ;print(i,'view: ',out2.size())
            out2 = out2.contiguous().view(B * T, sub_C, H, W)
            out2 = self.conv(out2)  # ;print(i,'out2: ',out2.size())

            out3_b = self.branch3_1(feat)  # ;print(i,'branch3: ',out3.size())
            out3 = (out3_b + out2_b + out1_b).contiguous().view(B, H, W, sub_C, T, 1).permute(0, 4, 5, 3, 1, 2)
            out3 = out3.contiguous().view(B * T, sub_C, H, W)
            out3 = self.conv(out3)  # ;print(i,'out3: ',out3.size())

            out4_b = self.branch4_1(feat)  # ;print(i,'branch4: ',out4.size())
            out4 = (out4_b + out3_b + out2_b + out1_b).contiguous().view(B, H, W, sub_C, T, 1).permute(0, 4, 5, 3, 1, 2)
            out4 = out4.contiguous().view(B * T, sub_C, H, W)
            out4 = self.conv(out4)  # ;print(i,'out4: ',out4.size())

            # feat0 = feat0.contiguous().view(B*T,sub_C, H, W)
            # out5 = self.branch5(feat0)#;print(i,'out5: ',out5.size())

            feat_list.append(torch.cat((out1, out2, out3, out4), 1))

        x = torch.cat(feat_list, 1)  # ;print('x.size() ',x.size()) # [32, 2560, 7, 7]
        # x = self.conv_all(x);print('conv_all ',x.size())
        # x = self.maxpool_s2(x)
        return x + residual



class TimeceptionBlockV2(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV2, self).__init__()

        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        self.branch1 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), groups=inter_channels)
        self.branch2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=inter_channels)
        self.branch3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=inter_channels)
        self.branch4 = nn.Conv2d(inter_channels, inter_channels, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        B, C, H, W = x.size()
        T = 8
        B = B // T

        x = self.conv1(x)
        x = x.view(-1, T, self.inter_channels, H, W)  # ;print('x.size() ',x.size())
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1)
        x = x.contiguous().view(B * H * W, self.inter_channels, T, 1)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = (out1+out2+out3+out4).contiguous().view(B, H, W, self.inter_channels, T, 1).permute(0, 4, 5, 3, 1, 2)
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)

        return residual + out

class TcpModule(nn.Module):
    def __init__(self, in_channels=128):
        super(TcpModule, self).__init__()

        self.branch1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), groups=in_channels)
        self.branch2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=in_channels)
        self.branch3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=in_channels)
        self.branch4_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=in_channels)

    def forward(self, x):
        out1_b = self.branch1_1(x)
        out2_b = self.branch2_1(x)
        out3_b = self.branch3_1(x)
        out4_b = self.branch4_1(x)
        out = torch.cat((out1_b, out2_b, out3_b, out4_b), 1)

        return out

class TimeceptionBlockV3(nn.Module):
    def __init__(self, channels, inter_channels=512, groups=4):
        super(TimeceptionBlockV3, self).__init__()

        self.inter_channels = inter_channels
        self.groups = groups

        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))

        self.tcp = TcpModule(in_channels=inter_channels // groups)

        self.conv_tcp = nn.Conv2d(inter_channels, inter_channels // groups, kernel_size=(1, 1), groups=4)

        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        B, C, H, W = x.size()
        T = 8
        B = B // T

        x = self.conv1(x)
        x = x.view(-1, T, self.inter_channels, H, W)  # ;print('x.size() ',x.size())
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1)
        x = x.contiguous().view(B * H * W, self.inter_channels, T, 1)

        _c = int(self.inter_channels / self.groups)
        xs = torch.split(x, split_size_or_sections=_c, dim=1)

        feat_list = []

        for i in range(self.groups):
            out = self.tcp(xs[i]);print('out ',out.size()) #[784, 1024, 8, 1]
            out = out.contiguous().view(B, H, W, self.inter_channels, T, 1).permute(0, 4, 5, 3, 1, 2)
            out = out.contiguous().view(B * T, self.inter_channels, H, W)
            out_s = torch.split(out, split_size_or_sections=_c, dim=1)
            out_s_list = list(out_s)
            out_s_list[1] = out_s_list[0] + out_s_list[1]
            out_s_list[2] = out_s_list[1] + out_s_list[2]
            out_s_list[3] = out_s_list[2] + out_s_list[3]
            out_s = torch.cat(out_s_list, 1)
            out_s = self.conv_tcp(out_s)
            feat_list.append(out_s)

        feat_list = torch.cat(feat_list, 1)
        out = self.conv2(feat_list)

        return residual + out


# class TimeceptionBlockV4(nn.Module):
#     def __init__(self, channels, intermed_channels=512, groups=4):
#         super(TimeceptionBlockV4, self).__init__()
#         self.conv3d_1 = nn.Conv3d(channels, intermed_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

#         self.temp_conv1 = nn.Conv3d(intermed_channels, intermed_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
#                                     padding=(1, 0, 0))
#         self.temp_conv2 = nn.Conv3d(intermed_channels, intermed_channels, kernel_size=(5, 1, 1), stride=(1, 1, 1),
#                                     dilation=(1, 1, 1), padding=(2, 0, 0))
#         self.temp_conv3 = nn.Conv3d(intermed_channels, intermed_channels, kernel_size=(7, 1, 1), stride=(1, 1, 1),
#                                     dilation=(1, 1, 1), padding=(3, 0, 0))
#         self.conv3d_2 = nn.Conv3d(intermed_channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

#     def forward(self, x):
#         #residual = x
#         BT, C, H, W = x.size()#;print('x ',x.size())
#         B = BT // 8 # Assume 8 frames for a video
#         T = 8
#         x = x.contiguous().view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # B, C, T, H, W

#         conv3d1 = self.conv3d_1(x)
#         temp_conv1 = self.temp_conv1(conv3d1)
#         temp_conv2 = self.temp_conv2(conv3d1)
#         temp_conv3 = self.temp_conv3(conv3d1)
#         out = self.conv3d_2(conv3d1 + temp_conv1 + temp_conv2 + temp_conv3)

#         out = out.permute(0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)
#         return out


class TimeceptionBlockV4_sf(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4_sf, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        self.slow = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.fast = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T
        x0 = self.conv1(x) # B, C, H, W

        x = x0.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out1 = self.fast(x)
        out1 = out1.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out1 = out1.contiguous().view(B * T, self.inter_channels, H, W)

        T = 2
        B = BT // T
        x = x0.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out2 = self.slow(x)
        out2 = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out2 = out2.contiguous().view(B * T, self.inter_channels, H, W)

        out = self.conv2(out1+out2)
        return residual + out


class TimeceptionBlockV4_sf(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4_sf, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        self.slow = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.fast = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.midd = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T
        x0 = self.conv1(x) # B, C, H, W

        x = x0.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out1 = self.fast(x)
        out1 = out1.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out1 = out1.contiguous().view(B * T, self.inter_channels, H, W)

        T = 2
        B = BT // T
        x = x0.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out2 = self.slow(x)
        out2 = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out2 = out2.contiguous().view(B * T, self.inter_channels, H, W)

        T = 4
        B = BT // T
        x = x0.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out3 = self.slow(x)
        out3 = out3.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out3 = out3.contiguous().view(B * T, self.inter_channels, H, W)

        out = self.conv2(out1+out2+out3)
        return residual + out



class TimeceptionBlockV4(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        #self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        #self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        #self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T

        x = self.conv1(x) # B, C, H, W
        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        #out1 = self.branch1(x)
        out2 = self.branch2(x)
        #out3 = self.branch3(x)
        #out4 = self.branch4(x)
        out = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)

        return residual + out


class TimeceptionBlockV4__st0(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4__st0, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        #self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        #self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        #self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

        self.conv_s = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.bn_s = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T

        x = self.conv1(x) # B, C, H, W

        x_s = self.conv_s(x)
        x_s = self.relu(self.bn_s(x_s))

        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out2 = self.branch2(x)
        out = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)

        out = self.conv2(out+x_s)

        return residual + out




class TimeceptionBlockV4__st(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4__st, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        #self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        #self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        #self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

        self.conv_s = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.bn_s = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T

        x = self.conv1(x) # B, C, H, W

        x_s = self.conv_s(x)
        x_s = self.relu(self.bn_s(x_s))

        x = x_s.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out2 = self.branch2(x)
        out = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)

        out = self.conv2(out)

        return residual + out


class TimeceptionBlockV4__ts(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4__ts, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        #self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        #self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        #self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

        self.conv_s = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.bn_s = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T

        x = self.conv1(x) # B, C, H, W

        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out2 = self.branch2(x)
        out = out2.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)

        x_s = self.conv_s(out)
        x_s = self.relu(self.bn_s(x_s))

        out = self.conv2(x_s)

        return residual + out





class TimeceptionBlockV4_1(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4_1, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)

        self.branch3_0 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.branch3_1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)

        self.branch4_0 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.branch4_1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        self.branch4_2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T

        x = self.conv1(x) # B, C, H, W
        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3_0(x)
        out3 = self.branch3_1(out3)
        out4 = self.branch4_0(x)
        out4 = self.branch4_1(out4)
        out4 = self.branch4_2(out4)
        out = (out1+out2+out3+out4).contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)

        return residual + out



class TimeceptionBlockV4_2(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV4_2, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))

        convs = []
        bns = []
        for i in range(4):
          convs.append(nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride = 1, padding=1, bias=False))
          bns.append(nn.BatchNorm1d(inter_channels))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        # self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        # self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        # self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T
        x = self.conv1(x) # B, C, H, W
        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = out.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)
        return residual + out




class SeparableConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv1d,self).__init__()
        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv1d(in_channels,out_channels,1,1,0,1,groups=1,bias=bias)
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class TimeceptionBlockV5(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV5, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        # self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        # self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        # self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        # self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.sepconv1 = SeparableConv1d(inter_channels, inter_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm1d(inter_channels)
        self.sepconv2 = SeparableConv1d(inter_channels, inter_channels, kernel_size = 3, padding = 1)
        self.conv = nn.Conv1d(inter_channels, inter_channels, kernel_size = 1, padding = 0)
        self.bn3 = nn.BatchNorm1d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T
        x = self.conv1(x) # B, C, H, W
        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T

        x = self.bn1(x)
        x2 = self.conv(x)
        x1 = self.sepconv1(x)
        x1 = F.relu(self.bn2(x1))
        x1 = self.sepconv2(x1)
        x = F.relu(self.bn3(x1 + x2))

        out = x.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)

        return residual + out


class mobileV2Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(mobileV2Conv,self).__init__()
        self.pw1 = nn.Conv1d(in_channels,in_channels,1,1,0,1,groups=1,bias=bias)
        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pw2 = nn.Conv1d(in_channels,out_channels,1,1,0,1,groups=1,bias=bias)
    def forward(self,x):
        #redisual = x
        x = self.pw1(x)
        x = self.conv1(x)
        x = self.pw2(x)
        return x#+redisual
class TimeceptionBlockV6(nn.Module):
    def __init__(self, channels, inter_channels=512):
        super(TimeceptionBlockV6, self).__init__()
        self.inter_channels = inter_channels
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=(1, 1))
        # self.branch1 = nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels)
        # self.branch2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=inter_channels)
        # self.branch3 = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2, groups=inter_channels)
        # self.branch4 = nn.Conv1d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3, groups=inter_channels)
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.sepconv1 = mobileV2Conv(inter_channels, inter_channels, kernel_size = 3, padding = 1)
        # self.bn2 = nn.BatchNorm1d(inter_channels)
        # self.sepconv2 = SeparableConv1d(inter_channels, inter_channels, kernel_size = 3, padding = 1)
        self.conv = nn.Conv1d(inter_channels, inter_channels, kernel_size = 1, padding = 0)
        self.bn3 = nn.BatchNorm1d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        BT, C, H, W = x.size()
        T = 8
        B = BT // T
        x = self.conv1(x) # B, C, H, W
        x = x.view(-1, T, self.inter_channels, H, W)  # B, T, C, H, W
        x = x.contiguous().view(B, T, self.inter_channels, H * W).permute(0, 3, 2, 1) # B, H*W, C, T
        x = x.contiguous().view(B * H * W, self.inter_channels, T) # B*H*W, C, T

        x = self.bn1(x)
        x2 = self.conv(x)
        x1 = self.sepconv1(x)
        # x1 = F.relu(self.bn2(x1))
        # x1 = self.sepconv2(x1)
        x = F.relu(self.bn3(x1 + x2))

        out = x.contiguous().view(B, H, W, self.inter_channels, T).permute(0, 4, 3, 1, 2)# B, H, W, C, T --> B, T, C, H, W
        out = out.contiguous().view(B * T, self.inter_channels, H, W)
        out = self.conv2(out)

        return residual + out


if __name__ == '__main__':
    import time
    data = torch.ones(32, 1024, 14, 14)

    model = TimeceptionBlock(channels=1024)
    model.eval()
    t1 = time.time()
    for i in range(10):
        out = model(data)
    t2 = time.time()
    print((t2-t1)/10.0)

#----------------------------------------------

    model2 = TimeceptionBlockV2(channels=1024)
    model2.eval()

    t1 = time.time()
    for i in range(10):
        out = model2(data)
    t2 = time.time()
    print((t2-t1)/10.0)

# ----------------------------------------------

    model3 = TimeceptionBlockV3(channels=1024, groups=4)
    model3.eval()

    t1 = time.time()
    for i in range(10):
        out = model3(data)
    t2 = time.time()
    print((t2-t1)/10.0)