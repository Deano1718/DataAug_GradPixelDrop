import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, **kwargs):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, nclass)

        self.multi_out = 0
        self.proto_layer = kwargs['proto_layer']
        self.proto_pool = kwargs['proto_pool']
        self.proto_norm = kwargs['proto_norm']
        if self.proto_pool == "max":
            self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        elif self.proto_pool == "ave":
            self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

    def define_proto(self,features):
        if self.proto_pool in ['max','ave']:
            features = self.proto_pool_f(features)
            
        if self.proto_norm:
            return F.normalize(features.view(features.shape[0],-1))
        else:
            return features.view(features.shape[0],-1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.proto_layer == 3:
            p = self.define_proto(out)
        out = self.layer4(out)
        if self.proto_layer == 4:
            p = self.define_proto(out)
        out = F.avg_pool2d(out, 4)
        if self.proto_layer == 5:
            p = self.define_proto(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if (self.multi_out):
            return p, out
        else:
            return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, HW, stride=1):
        super(BasicBlock, self).__init__()

        #self.kwargs = kwargs
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn2 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn2 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, HW, stride=1):
        super(BasicBlockLayer, self).__init__()

        #self.kwargs = kwargs
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.LayerNorm([planes,HW,HW], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.LayerNorm([planes,HW,HW], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn2 = nn.BatchNorm2d(planes)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn2 = nn.LayerNorm([planes,2048//planes,2048//planes], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.LayerNorm([self.expansion * planes,HW,HW], elementwise_affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, channels=3, bn=1, **kwargs):
        super(ResNet, self).__init__()
        #self.kwargs = kwargs
        self.in_planes = int(64 * scale)
        self.channels = channels
        print (self.in_planes)

        if bn:
            self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(int(64*scale))
        else:
            self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.LayerNorm([int(64*scale),32,32], elementwise_affine=False)

        # if self.kwargs['norm_type'] == 'batch':
        #     self.bn1 = nn.BatchNorm2d(64)
        # elif self.kwargs['norm_type'] == 'layer':
        #     self.bn1 = nn.LayerNorm([self.in_planes,32,32], elementwise_affine=True)
        # elif self.kwargs['norm_type'] == 'instance':
        #     self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, int(64 * scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(128 * scale), num_blocks[1], HW=32, stride=2)
        self.layer3 = self._make_layer(block, int(256 * scale), num_blocks[2], HW=16, stride=2)
        self.layer4 = self._make_layer(block, int(512 * scale), num_blocks[3], HW=8, stride=2)
        self.linear = nn.Linear(int(512 * scale) * block.expansion, nclass)

        self.multi_out = 0


            
    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
            
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
    
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        
        return out

class ResNetIN(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, **kwargs):
        super(ResNetIN, self).__init__()
        #self.kwargs = kwargs        
        self.in_planes = 64 // scale

        self.conv1 = nn.Conv2d(3, 64 // scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // scale)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64//scale, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//scale, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//scale, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//scale, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512//scale * block.expansion, nclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.multi_out = 0
        self.proto_layer = kwargs['proto_layer']
        self.proto_pool = kwargs['proto_pool']
        self.proto_norm = kwargs['proto_norm']

        
        if self.proto_pool == "max":
            self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        elif self.proto_pool == "ave":
            self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

    def define_proto(self,features):
        if self.proto_pool in ['max','ave']:
            features = self.proto_pool_f(features)

        #if self.proto_norm:
        #    return F.normalize(features.view(features.shape[0],-1))
        #else:
        return features.view(features.shape[0],-1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.proto_layer == 3:
            p = self.define_proto(out)
            
        out = self.layer4(out)
        
        if self.proto_layer == 4:
            p = self.define_proto(out)
            
        #out = self.avgpool(out)

        if self.proto_pool == 'ave':
            out = F.avg_pool2d(out, 4)

        if self.proto_pool == 'max':
            out = F.max_pool2d(out, 4)

        
        #if self.proto_layer == 5:
        #    p = self.define_proto(out)
            
        out = out.view(out.size(0), -1)

        if self.proto_norm:
            out = F.normalize(out)

        
        out = self.linear(out)
        
        if (self.multi_out):
            return p, out
        else:
            return out

class ResNetTiny(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, channels=3, bn=1, **kwargs):
        super(ResNetTiny, self).__init__()
        #self.kwargs = kwargs        
        self.in_planes = int(64 * scale)
        self.channels = channels

        #self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(int(64 * scale))

        if bn:
            self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(int(64*scale))
        else:
            self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=7, stride=2, padding=3, bias=True)
            self.bn1 = nn.LayerNorm([int(64*scale),32,32], elementwise_affine=False)

        self.layer1 = self._make_layer(block, int(64*scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(128*scale), num_blocks[1], HW=32, stride=2)
        self.layer3 = self._make_layer(block, int(256*scale), num_blocks[2], HW=16, stride=2)
        self.layer4 = self._make_layer(block, int(512*scale), num_blocks[3], HW=8, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512*scale) * block.expansion, nclass)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

        self.multi_out = 0
        self.proto_layer = kwargs['proto_layer']
        self.proto_pool = kwargs['proto_pool']
        self.proto_norm = kwargs['proto_norm']

        
        if self.proto_pool == "max":
            self.proto_pool_f = nn.AdaptiveMaxPool2d((1,1))
        elif self.proto_pool == "ave":
            self.proto_pool_f = nn.AdaptiveAvgPool2d((1,1))

    def define_proto(self,features):
        if self.proto_pool in ['max','ave']:
            features = self.proto_pool_f(features)

        #if self.proto_norm:
        #    return F.normalize(features.view(features.shape[0],-1))
        #else:
        return features.view(features.shape[0],-1)


    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.proto_layer == 3:
            p = self.define_proto(out)
            
        out = self.layer4(out)
        
        if self.proto_layer == 4:
            p = self.define_proto(out)
            
        #out = self.avgpool(out)

        if self.proto_pool == 'ave':
            out = F.avg_pool2d(out, 4)

        if self.proto_pool == 'max':
            out = F.max_pool2d(out, 4)

        
        #if self.proto_layer == 5:
        #    p = self.define_proto(out)
            
        out = out.view(out.size(0), -1)

        if self.proto_norm:
            out = F.normalize(out)

        
        out = self.linear(out)
        
        if (self.multi_out):
            return p, out
        else:
            return out


def PreActResNet18(nclass, **kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], nclass, **kwargs)

def ResNet18(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, 1, **kwargs)

def ResNet18Tiny(nclass, scale, channels, **kwargs):
    return ResNetTiny(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, 1, **kwargs)

def ResNet18L(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlockLayer, [2, 2, 2, 2], nclass, scale, channels, 0, **kwargs)

def ResNet18TinyL(nclass, scale, channels, **kwargs):
    return ResNetTiny(BasicBlockLayer, [2, 2, 2, 2], nclass, scale, channels, 0, **kwargs)


def ResNet18IN(nclass, scale, **kwargs):
    return ResNetIN(BasicBlock, [2, 2, 2, 2], nclass, scale, **kwargs)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
