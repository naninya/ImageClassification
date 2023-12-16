import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, device, image_size, n_classes):
        super(LinearModel, self).__init__()
        self.fcn = nn.Linear(image_size[0]*image_size[1], n_classes)
        self.to(device)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = self.fcn(x)
        return x

class SimpleConvModel(nn.Module):
    def __init__(self, device, image_size, n_classes):
        super(SimpleConvModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="same")
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResnetModel(nn.Module):
    def __init__(self, device, image_size, n_classes):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        self.model = resnet
        self.dropout = nn.Dropout(0.5)
        self.layer1 = nn.Linear(1000, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, n_classes)
        self.to(device)

    def forward(self, batch):
        if batch.size(1) == 1:
            outputs = torch.tile(batch, dims=(1,3,1,1))
        outputs = self.model(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer1(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer3(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs

class ResnetImageNetModel(nn.Module):
    def __init__(self, device, image_size, n_classes):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.model = resnet
        self.dropout = nn.Dropout(0.5)
        self.layer1 = nn.Linear(1000, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, n_classes)
        self.to(device)

    def forward(self, batch):
        if batch.size(1) == 1:
            outputs = torch.tile(batch, dims=(1,3,1,1))
        outputs = self.model(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer1(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer3(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs
    
    
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetResnet50(nn.Module):
    DEPTH = 6

    def __init__(self, device, image_size, n_classes=10):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)
        
        # customize output
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.custom_out = nn.Linear(10*32*32, n_classes)
        
        self.to(device)

    def forward(self, x, with_output_feature_map=False):
        x = torchvision.transforms.Resize(32)(x)
        x = torch.tile(x, dims=(1,3,1,1))
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetResnet50.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetResnet50.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        x = self.flatten(x)
        x = self.custom_out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

# model = UNetResnet50(device).cuda()
# inp = torch.rand((2, 1, 28, 28)).cuda()
# out = model(inp)