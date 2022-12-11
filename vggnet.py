import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.GroupNorm(num_groups=16, num_channels=num_channels, eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

class MyBatchNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyBatchNorm, self).__init__()
        ## change num_groups to 32
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True, norm_layer=MyGroupNorm):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class VGG(nn.Module):

    def __init__(self, n_blocks, norm_layer: Optional[Callable[..., nn.Module]] = None, num_classes = 10):
        super(VGG, self).__init__()
        self.n_blocks = n_blocks
        self.inplanes = 64
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.middle_fc1 = nn.Linear(64 , num_classes)
        # self.scala1 = nn.Sequential(
        #     SepConv(
        #         channel_in=64,
        #         channel_out=64,
        #         norm_layer=norm_layer
        #     ),
        #     SepConv(
        #         channel_in=64,
        #         channel_out=64,
        #         norm_layer=norm_layer
        #     ),

        #     nn.AvgPool2d(4, 4)
        # )
        # self.attention1 = nn.Sequential(
        #     SepConv(
        #         channel_in=64,
        #         channel_out=64,
        #         norm_layer=norm_layer
        #     ),
        #     norm_layer(64),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Sigmoid()
        # )

        if n_blocks > 1:
            
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = norm_layer(128)
            self.relu = nn.ReLU(inplace=True)
        
            # self.middle_fc2 = nn.Linear(128, num_classes)
            # self.scala2 = nn.Sequential(
            #     SepConv(
            #         channel_in=128,
            #         channel_out=128,
            #         norm_layer=norm_layer
            #     ),
            #     nn.AvgPool2d(4, 4)
            # )
            # self.attention2 = nn.Sequential(
            #     SepConv(
            #         channel_in=128,
            #         channel_out=128,
            #         norm_layer=norm_layer
            #     ),
            #     norm_layer(128),
            #     nn.ReLU(),
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            #     nn.Sigmoid()
            # )
        

        if n_blocks > 2:
            
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = norm_layer(256)
            self.relu = nn.ReLU(inplace=True)

            # self.middle_fc3 = nn.Linear(256, num_classes)
            # self.scala3 = nn.Sequential(
            #     SepConv(
            #         channel_in=256,
            #         channel_out=256,
            #         norm_layer=norm_layer
            #     ),
            #     nn.AvgPool2d(2, 2)
            # )
            # self.attention3 = nn.Sequential(
            #     SepConv(
            #         channel_in=256,
            #         channel_out=256,
            #         norm_layer=norm_layer
            #     ),
            #     norm_layer(256),
            #     nn.ReLU(),
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            #     nn.Sigmoid()
            # )


        if n_blocks > 3:
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = norm_layer(512)
            self.relu = nn.ReLU(inplace=True)
            self.scala = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # fea1 = self.attention1(x)
        # fea1 = fea1 * x
        # out1_feature = self.scala1(fea1).view(x.size(0), -1)
        # middle_output1 = self.middle_fc1(out1_feature)

        # if self.n_blocks == 1:
        #     return [middle_output1], [out1_feature]

        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # fea2 = self.attention2(x)
        # fea2 = fea2 * x
        # out2_feature = self.scala2(fea2).view(x.size(0), -1)
        # middle_output2 = self.middle_fc2(out2_feature)

        # if self.n_blocks == 2:
        #     return [middle_output1, middle_output2], [out1_feature, out2_feature]

        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # fea3 = self.attention3(x)
        # fea3 = fea3 * x
        # out3_feature = self.scala3(fea3).view(x.size(0), -1)
        # middle_output3 = self.middle_fc3(out3_feature)

        # if self.n_blocks == 3:
        #     return [middle_output1, middle_output2, middle_output3], [out1_feature, out2_feature, out3_feature]

        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.scala(x)

        x = x.view(x.size(0), -1)
        output4 = self.fc1(x)
    
        #return [middle_output1, middle_output2, middle_output3, output4], [None, None, None, None]
        return output4

def make_VGG(n_blocks=4, norm='gn'):
    if norm == 'gn':
        norm_layer = MyGroupNorm
        
    elif norm == 'bn':
        norm_layer = MyBatchNorm

    return VGG(n_blocks, norm_layer=norm_layer)
    

# if __name__ == "__main__":
#     from ptflops import get_model_complexity_info

#     model = make_VGG(n_blocks=4, norm='bn')

#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
#                                                 print_per_layer_stat=False, verbose=True, units='MMac')

#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))

