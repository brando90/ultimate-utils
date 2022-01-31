"""

ref:
    - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/models/convnet.py
"""

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    NOTE: this is missing the maxpool2d from the opt as few shot arch.
    """

    def __init__(self, num_classes=-1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        #             ('pool4', nn.MaxPool2d(kernel_size=2))]))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(64, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False):
        out = self.layer1(x)
        f0 = out
        out = self.layer2(out)
        f1 = out
        out = self.layer3(out)
        f2 = out
        out = self.layer4(out)
        f3 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat = out

        if self.num_classes > 0:
            out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, feat], out
        else:
            return out


def convnet4(**kwargs):
    """Four layer ConvNet
    """
    model = ConvNet(**kwargs)
    return model


# -- tests

def original_rfs_test_ala_mi():
    model = convnet4(num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    feat, logit = model(data, is_feat=True)
    print(feat[-1].shape)
    print(logit.shape)


def pass_cifarfs_data_through_5cnn_model_test():
    """
    shape is torch.Size([3, 32, 32])
    :return:
    """
    # args = Namespace()
    # args.data_root = Path('~/data/CIFAR-FS/').expanduser()
    # args.data_aug = True
    # imagenet = CIFAR100(args.data_root, args.data_aug, 'train')
    # print(len(imagenet))
    # print(imagenet.__getitem__(500)[0].shape)

    B = 4
    CHW = [3, 32, 32]
    x = torch.randn([B] + CHW)

    model = convnet4(num_classes=5)
    feat, logit = model(x, is_feat=True)
    # print(feat[-1].shape)
    [print(feature.shape) for feature in feat]
    print(logit.shape)


if __name__ == '__main__':
    pass_cifarfs_data_through_5cnn_model_test()
    print('Done\a')
