"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        # print(output.size())
        output = self.conv4_x(output)
        # print(output.size())
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(block=BasicBlock, num_block=[2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(block=BasicBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(block=BottleNeck, num_block=[3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    """ return a ResNet 101 object
    """
    return ResNet(block=BottleNeck, num_block=[3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    """ return a ResNet 152 object
    """
    return ResNet(block=BottleNeck, num_block=[3, 8, 36, 3], num_classes=num_classes)


class SimpleCNN(nn.Module):
    """simple CNN model
    """
    def __init__(self, in_channels=1, out_channels=256, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * 1 * 1, num_classes)
        self.adptavgpool1 = nn.AdaptiveAvgPool2d(14)
        self.adptavgpool2 = nn.AdaptiveAvgPool2d(7)
        self.adptavgpool3 = nn.AdaptiveAvgPool2d(3)
        self.adptavgpool4 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.adptavgpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.adptavgpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.adptavgpool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.adptavgpool4(x)
        x = self.flatten(x)
        # print(x.size())
        y = self.fc(x)
        return y


"""conv vae in pytorch
ref :https://github.com/sksq96/pytorch-vae 
"""


class VAE_Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VAE_UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


# CNN VAE model
class CNN_VAE(nn.Module):
    """Conv + VAE.
    """

    def __init__(self, in_channels=512, h_dim=512, z_dim=32):
        super(CNN_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            VAE_Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            VAE_UnFlatten(),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, in_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        # print("h before bottleneck", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        # print("x before encdoer", x.size())
        h = self.encoder(x)
        # print("h after encoder", h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print("z after encoder", z.size())
        z = self.decode(z)
        # print("z after decoder", z.size())
        return z, mu, logvar


# CNN AE model
class CNN_AE(nn.Module):
    """MaxPool is added
    """
    def __init__(self, in_channels=256):
        super(CNN_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(int(in_channels / 2), int(in_channels / 4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(int(in_channels / 4), int(in_channels / 8), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid(),  # is it needed?
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(in_channels / 8), int(in_channels / 4), kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 4), int(in_channels / 2), kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 2), in_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print("x after encoder", x.size())
        y = self.decoder(x)
        # print("y after decoder", y.size())
        return y


class SimpleNN(nn.Module):  # nn.Linear in_channel -> out_channels(class)
    """Simple neural net, only one FC layer.
    """

    def __init__(self, in_channels=64 * 5 * 5, out_channels=10):
        super(SimpleNN, self).__init__()
        self.FC = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # print("SimpleNN forward, x size:"+str(x.size()))
        y = self.FC(x)
        return y


class Identity(nn.Module):
    """To remove layers from model, used as dummy
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG16(nn.Module):
    """VGG16, Linear in_out channel has been modified to train image, sized 128x128
    """

    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        # print(x.size())
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)

        x = x.view(x.shape[0], -1)
        # print(x.size())

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)

        y = self.fc3(x)
        return y


class CNN_AE_legacy(nn.Module):
    """
    legacy, channel up-scaleing cost is too heavy
    and it's loss is too high to say model is trained the feature map
    +inverted bottleneck structure, channel dimension increased(which is exact opposite of original intention)
    would say that it's garbage model codes,,,
    """

    def __init__(self, in_channels=256):
        super(CNN_AE_legacy, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 768, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print("x after encoder", x.size())
        y = self.decoder(x)
        # print("y after decoder", y.size())
        return y
    # CNN AE model


class CNN_AE_JJinmak(nn.Module):
    """avgpool precede
    """

    def __init__(self, in_channels=512):
        super(CNN_AE_JJinmak, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 2), int(in_channels / 4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 4), int(in_channels / 8), kernel_size=4, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(in_channels / 8), int(in_channels / 4), kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 4), int(in_channels / 2), kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 2), in_channels, kernel_size=2, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print("x after encoder", x.size())
        y = self.decoder(x)
        # print("y after decoder", y.size())
        return y

class Encoder(nn.Module):    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        #print(x.size())
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 4 * 4),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class conv_ae(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim=encoded_space_dim, fc2_input_dim=fc2_input_dim)
        self.decoder = Decoder(encoded_space_dim=encoded_space_dim, fc2_input_dim=fc2_input_dim)

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y



class Encoder_F(nn.Module):    
    def __init__(self, encoded_space_dim, in_channel):
        super().__init__()
        ### Convolutional section
        in_channel = int(in_channel)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel//2, 1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channel//2, in_channel//2, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//2, in_channel//2, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//2, in_channel//4, 1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channel//4, in_channel//4, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//4, in_channel//4, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//4, in_channel//8, 1, stride=1),
            nn.BatchNorm2d(in_channel//8),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channel//8, in_channel//8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//8, in_channel//8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//8, in_channel//16, 1, stride=1),
            nn.BatchNorm2d(in_channel//16),
            nn.ReLU(True),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, encoded_space_dim),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder_F(nn.Module):
    def __init__(self, encoded_space_dim, in_channel):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 7 * 7),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel//16, in_channel//8, 1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//8, in_channel//8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//8, in_channel//8, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel//8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channel//8, in_channel//4, 1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//4, in_channel//4, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//4, in_channel//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channel//4, in_channel//2, 1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//2, in_channel//2, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//2, in_channel//2, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channel//2, in_channel, 1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = torch.sigmoid(x)
        return x

class conv_ae_F(nn.Module):
    def __init__(self, encoded_space_dim, in_channel):
        super().__init__()
        self.encoder = Encoder_F(encoded_space_dim=encoded_space_dim, in_channel=in_channel)
        self.decoder = Decoder_F(encoded_space_dim=encoded_space_dim, in_channel=in_channel)
    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y



class Encoder_JJJmak(nn.Module):    
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1024, 128, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * 4 * 4, encoded_space_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        #print(x.size())
        x = self.encoder_lin(x)
        return x

class Decoder_JJJmak(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64 * 4 * 4),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(64, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, 6, stride=2, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1024, 6, stride=2, padding=2),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class conv_ae_JJJmak(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder = Encoder_JJJmak(encoded_space_dim=encoded_space_dim)
        self.decoder = Decoder_JJJmak(encoded_space_dim=encoded_space_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        y = self.decoder(x)
        return y
