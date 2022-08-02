import os
import gc
import numpy as np
import cv2
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from multiprocessing import Manager

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class MyHRNet(nn.Module):
    def __init__(self, in_channels=3):
        self.inplanes = 64
        extra = {
            "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2',
                                  'stage3', 'transition3', 'stage4'],
            "FINAL_CONV_KERNEL": 1,
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [32, 64],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [32, 64, 128],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [32, 64, 128, 256],
                "FUSE_METHOD": 'SUM'
            }
        }
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        feats = self.stage4(x_list)[0]  # B, 32, H // 4, W // 4

        feats = self.final_layer(feats)
        return feats

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))


class MySelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.m_qv = nn.Linear(dim, dim * 2, bias=True)
        self.m_k = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, flow):
        b, c, h, w = x.shape
        n = h * w
        x = x.permute(0, 2, 3, 1).contiguous().view(b, n, c)
        flow = flow.permute(0, 2, 3, 1).contiguous().view(b, n, c)

        shortcut = x
        x = self.norm1(x)
        flow = self.norm1(flow)

        qv = self.m_qv(x).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, v = qv[0], qv[1]

        k = self.m_k(flow).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x


class MotionNet_v21(nn.Module):
    def __init__(self, train_size=512):
        super().__init__()
        self.name = self.__class__.__name__
        self.train_size = train_size

        TRIANGLE_SIZE = 80
        self.TRIANGLE_INDEX = []
        for i in range(TRIANGLE_SIZE):
            for j in range(TRIANGLE_SIZE - i):
                self.TRIANGLE_INDEX.append([i, j])
        self.TRIANGLE_INDEX = torch.FloatTensor(self.TRIANGLE_INDEX).unsqueeze(0)
        self.TRIANGLE_VERTEX = torch.FloatTensor([[0, 0], [0, TRIANGLE_SIZE], [TRIANGLE_SIZE, 0]]).unsqueeze(0)
        self.small_triangle = torch.FloatTensor([[[0, 0], [0, 1], [1, 0]]])
        self.TRIANGLE_dict = Manager().dict()
        self.device = 'cuda:0'

        self.face_triangle_list = [
            [0, 1, 41],
            [0, 17, 36],
            [0, 36, 41],
            [1, 2, 29],
            [1, 29, 41],
            [2, 3, 31],
            [2, 29, 30],
            [2, 30, 31],
            [3, 4, 48],
            [3, 31, 48],
            [4, 5, 48],
            [5, 6, 48],
            [6, 7, 59],
            [6, 48, 59],
            [7, 57, 59],
            [7, 8, 57],
            [8, 9, 57],
            [9, 55, 57],
            [9, 10, 55],
            [10, 11, 54],
            [10, 54, 55],
            [11, 12, 54],
            [12, 13, 54],
            [13, 14, 35],
            [13, 35, 54],
            [14, 15, 29],
            [14, 29, 30],
            [14, 30, 35],
            [15, 16, 45],
            [15, 29, 46],
            [15, 45, 46],
            [16, 26, 45],
            [17, 18, 19],
            [17, 19, 37],
            [17, 36, 37],
            [19, 20, 38],
            [19, 37, 38],
            [19, 20, 38],
            [19, 20, 24],
            [20, 21, 23],
            [20, 21, 38],
            [21, 22, 23],
            [21, 22, 27],
            [21, 27, 39],
            [21, 38, 39],
            [22, 23, 43],
            [22, 27, 42],
            [22, 42, 43],
            [23, 24, 25],
            [23, 25, 43],
            [25, 26, 44],
            [25, 43, 44],
            [26, 44, 45],
            [27, 28, 39],
            [27, 28, 42],
            [28, 29, 40],
            [28, 29, 47],
            [28, 39, 40],
            [28, 42, 47],
            [29, 40, 41],
            [29, 46, 47],
            [30, 31, 32],
            [30, 32, 33],
            [30, 33, 34],
            [30, 34, 35],
            [31, 48, 49],
            [31, 32, 49],
            [32, 33, 49],
            [33, 34, 53],
            [33, 49, 51],
            [33, 51, 53],
            [34, 35, 53],
            [35, 53, 54],
            [36, 37, 41],
            [37, 38, 41],
            [38, 39, 40],
            [38, 40, 41],
            [42, 43, 47],
            [43, 44, 47],
            [44, 45, 46],
            [44, 46, 47],
            [48, 49, 59],
            [49, 51, 66],
            [49, 59, 66],
            [53, 54, 55],
            [51, 53, 66],
            [53, 55, 66],
            [55, 57, 66],
            [57, 59, 66]
        ]

        self.encoder_rgb = MyHRNet()
        self.encoder_motion = MyHRNet(in_channels=2)
        self.sa = MySelfAttention(512, 512 // 32)

        self.feats2mask = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1),
        )
        self.feats2cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )

    def trianglePoints_BatchEstimate(self, pts):
        num, _, _ = pts.shape
        if num in self.TRIANGLE_dict.keys():
            tri_pts, warp_points, batch_index = self.TRIANGLE_dict[num]
        else:
            tri_pts = self.TRIANGLE_VERTEX.repeat(num, 1, 1)
            warp_points = self.TRIANGLE_INDEX.repeat(num, 1, 1)
            batch_index = [[int(bi / len(self.face_triangle_list))] * self.TRIANGLE_INDEX.shape[1] for bi in range(num)]
            batch_index = torch.tensor(batch_index).view((num, self.TRIANGLE_INDEX.shape[1], 1))
            self.TRIANGLE_dict.update({num: [tri_pts, warp_points, batch_index]})
        tri_pts = tri_pts.to(self.device)
        warp_points = warp_points.to(self.device)
        batch_index = batch_index.to(self.device)

        M1, M2 = kornia.geometry.transform.get_tps_transform(tri_pts, pts)
        dense_pts = kornia.geometry.transform.warp_points_tps(warp_points, pts, M1, M2)
        dense_pts = torch.round(dense_pts)
        dense_pts[torch.any((dense_pts < 0) | (dense_pts >= self.train_size), dim=2)] = 0
        dense_pts = torch.cat([dense_pts, batch_index], dim=2)

        del tri_pts, warp_points, batch_index
        gc.collect()
        return dense_pts

    def flow_estimation(self, buffer, marks):
        buffer = buffer.to(self.device)
        marks = marks.to(self.device)
        tmp = marks[:, self.face_triangle_list]
        pts1 = tmp[:-1, :, :, :].view(-1, 3, 2)
        pts2 = tmp[1:, :, :, :].view(-1, 3, 2)
        square_pts1 = torch.cat([pts1, torch.ones([pts1.shape[0], 3, 1]).to(self.device)], dim=2)
        det = torch.abs(torch.linalg.det(square_pts1))
        det_shift = torch.cat([det[1:], torch.tensor([1e4]).to(self.device)])
        index = (det == 0) | (det_shift == 0)
        pts1[index] = self.small_triangle.clone().to(self.device)
        pts2[index] = self.small_triangle.clone().to(self.device)
        M1, M2 = kornia.geometry.transform.get_tps_transform(pts1, pts2)
        points = self.trianglePoints_BatchEstimate(pts1)
        warp_points = points[:, :, :2]
        next_points = kornia.geometry.transform.warp_points_tps(warp_points, pts2, M1, M2)
        next_points[warp_points == 0] = 0
        uv_vector = next_points - warp_points
        uv_vector = torch.clip(uv_vector, -buffer.shape[2], buffer.shape[2]).float()
        uv_vector = uv_vector.view((-1, 2))
        points = points.contiguous().view((-1, 3)).transpose(0, 1).long()
        buffer.index_put_((points[2], points[1], points[0]), uv_vector)
        del tmp, pts1, pts2, square_pts1, det, det_shift, index, M1, M2, points, warp_points, next_points, uv_vector
        gc.collect()
        return buffer.detach().cuda()

    def flow_initialization(self, buffer, marks):
        try:
            amm = self.flow_estimation(buffer, marks)
        except:
            amm = buffer
        amm = amm.permute(0, 3, 1, 2)
        return amm

    def forward(self, face, buffer, marks):
        B, V, C, H, W = face.shape
        face = face.view(B * V, C, H, W)
        buffer = buffer.view(B * V, H, W, 2)
        marks = marks.view(B * V, 68, 2)

        # Anchor-Mesh Motion
        amm = self.flow_initialization(buffer, marks)
        amm = amm.permute(1, 0, 2, 3)
        amm = amm.reshape(2, -1)
        f_std, f_mean = torch.std_mean(amm, dim=1)
        amm = amm.reshape(B * V, 2, H, W)
        if torch.count_nonzero(f_std) == 2:
            norm = transforms.Normalize(f_mean, f_std)
            amm = norm(amm)
        amm = amm.reshape(B * V, 2, H, W)

        # Feature encoding
        x_f = self.encoder_rgb(face)
        x_m = self.encoder_motion(amm)
        x_a = self.sa(x_f, x_m)

        # Localization branch
        localization = self.feats2mask(x_a)
        localization = localization.view(B, V, 2, H, W)

        # Classification branch
        classification = F.adaptive_avg_pool2d(x_a, (1, 1))
        classification = self.feats2cls(classification.view(classification.size(0), -1))
        classification = torch.mean(classification.view(B, V, -1), dim=1)

        return classification, localization
