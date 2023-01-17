"""
Adapted from: https://github.com/WangYueFt/rfs
"""
from argparse import Namespace

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = x.device
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(device)
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = mask.device

        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().to(device)
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).to(device).long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.cls = nn.Linear(640, self.num_classes)
    #         self.classifier = self.cls
    #
    # @property
    # def classifier(self):
    #     assert self.cls is self.classifier, f'The classifier and cls layer should be the same object!'
    #     return self.cls
    #
    # @classifier.setter
    # def classifier(self, new_classifier):
    #     assert self.cls is self.classifier, f'The classifier and cls layer should be the same object!'
    #     self.cls = new_classifier
    #     self.classifier = new_classifier
    #     assert self.cls is self.classifier, f'The classifier and cls layer should be the same object!'

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.cls(x)

        if is_feat:
            return [f0, f1, f2, f3, feat], x
        else:
            return x

    def get_embedding(self, x):
        [f0, f1, f2, f3, feat], x = self.forward(x, is_feat=True)
        return feat


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    indeed, only (1 + 1 + 1 + 1) * 3 + 1 = 12 + 1 layers

    note:
        - each block has 3 conv layers, so 1,1,1,1 blocks has 3,3,3,3 conv layers so 12 layers. When it says its
        a resnet 12 it means it has 12 conv layers.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    indeed, only (1 + 1 + 2 + 2) * 3 + 1 = 19 layers
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 48+1 = 49 layers

    note: it doesn't seem to be consistent with their own couting for resnet12. Sometimes they count the final layer
    sometimes they dont't.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def resnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def seresnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet18(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet24(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-24 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet50(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-50 model.
    indeed, only (3 + 4 + 6 + 3) * 3 + 1 = 49 layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


def seresnet101(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-101 model.
    indeed, only (3 + 4 + 23 + 3) * 3 + 1 = 100 layers
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], keep_prob=keep_prob, avg_pool=avg_pool, use_se=True, **kwargs)
    return model


model_dict = {
    'resnet12_rfs': resnet12,
    'resnet12_rfs_mi': resnet12,
    'resnet12_hdb1_mio': resnet12,
    'resnet12_rfs_cifarfs_fc100': resnet12,

    'resnet18_rfs': resnet18,
    'resnet24_rfs': resnet24,
    'resnet50_rfs': resnet50,
    'resnet101_rfs': resnet101,
    'seresnet12_rfs': seresnet12,
    'seresnet18_rfs': seresnet18,
    'seresnet24_rfs': seresnet24,
    'seresnet50_rfs': seresnet50,
    'seresnet101_rfs': seresnet101,
}


def get_resnet_rfs_model_mi(model_opt: str,
                            avg_pool=True,
                            drop_rate=0.1,
                            dropblock_size=5,
                            num_classes=64,
                            ) -> tuple[nn.Module, dict]:
    """
    Get resnet_rfs model according to the string model_opt
        e.g. model_opt = resnet12

    ref:
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/models/util.py#L7
    """
    model_hps: dict = {'avg_pool': avg_pool,
                       'drop_rate': drop_rate,
                       'dropblock_size': dropblock_size,
                       'num_classes': num_classes}
    model: nn.Module = model_dict[model_opt](avg_pool=avg_pool,
                                             drop_rate=drop_rate,
                                             dropblock_size=dropblock_size,
                                             num_classes=num_classes)
    return model, model_hps


def get_resnet_rfs_model_cifarfs_fc100(model_opt: str,
                                       num_classes,
                                       avg_pool=True,
                                       drop_rate=0.1,
                                       dropblock_size=2,
                                       ) -> tuple[nn.Module, dict]:
    """
    ref:
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/models/util.py#L23
    """
    model_hps: dict = {'avg_pool': avg_pool,
                       'drop_rate': drop_rate,
                       'dropblock_size': dropblock_size,
                       'num_classes': num_classes}
    model: nn.Module = model_dict[model_opt](avg_pool=avg_pool,
                                             drop_rate=drop_rate,
                                             dropblock_size=dropblock_size,
                                             num_classes=num_classes)
    return model, model_hps

def get_recommended_batch_size_mi_resnet12rfs_body(safety_margin: int = 10):
    """
    Loop through all the layers and computing the largest B recommnded. Most likely the H*W that is
    smallest will win but formally just compute B_l for each layer that your computing sims/dists and then choose
    the largest B_l. That ceil(B_l) satisfies B*H*W >= s*C for all l since it's the largest.

        recommended_meta_batch_size = ceil( max([s*C_l/H_l*W_l for l in 1...L]) )

    but really it's better just to choose one layer and do it for that layer. I recommend rep layer.

    Note: if the cls is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 =
        s_cls = B/n_c = 26/5 =
    """
    # todo - WARNING: these numbers are for MI and 5CNN
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 13
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 26
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


def get_recommended_batch_size_mi_resnet12rfs_head(safety_margin: int = 10):
    """
    The cls/head is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 = 2.6
        s_cls = B/n_c = 26/5 = 5.2
    note:
        - for meta-learning we have [B, M, n_cls] and the meta-batch dim is not part of this calculation
        since the distance is calculate per task (i.e. per [M, n_cls] layer_matrix). So we use M in the
        previous reasoning for B. So we do:
            s(M, n_cls) = M/n_cls
        and we want to choose M s.t. s(M, n_cls) >= 5, 10 or 20.
    """
    # todo - WARNING: these numbers are for MI and 5CNN
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 50
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 100
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


def get_recommended_batch_size_cifarfs_resnet12rfs_body(safety_margin: int = 10):
    """
    Loop through all the layers and computing the largest B recommnded. Most likely the H*W that is
    smallest will win but formally just compute B_l for each layer that your computing sims/dists and then choose
    the largest B_l. That ceil(B_l) satisfies B*H*W >= s*C for all l since it's the largest.

        recommended_meta_batch_size = ceil( max([s*C_l/H_l*W_l for l in 1...L]) )

    but really it's better just to choose one layer and do it for that layer. I recommend rep layer.

    Note: if the cls is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 =
        s_cls = B/n_c = 26/5 =
    """
    # todo - WARNING: these numbers are for MI and 5CNN
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 13
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 26
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


def get_recommended_batch_size_cifarfs_resnet12rfs_head(safety_margin: int = 10):
    """
    The cls/head is present then we need B >= s*D since the output for it has shape
    [B, n_c] where n_c so we need, B >= 10*5 = 50 for example.
    s being used for B = 13 is
        s_cls = B/n_c = 13/5 = 2.6
        s_cls = B/n_c = 26/5 = 5.2
    note:
        - for meta-learning we have [B, M, n_cls] and the meta-batch dim is not part of this calculation
        since the distance is calculate per task (i.e. per [M, n_cls] layer_matrix). So we use M in the
        previous reasoning for B. So we do:
            s(M, n_cls) = M/n_cls
        and we want to choose M s.t. s(M, n_cls) >= 5, 10 or 20.
    """
    # todo - WARNING: these numbers are for MI and 5CNN
    if safety_margin == 10:
        # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
        return 50
    elif safety_margin == 20:
        # -- satisfies B >= (20*32)/(5**2) = 25.6 for this specific 5CNN model
        return 100
    else:
        raise ValueError(f'Not implemented for value: {safety_margin=}')


def get_feature_extractor_conv_layers(L: int = 4, include_cls: bool = False) -> list[str]:
    """
    how to deal with: getattr(args.model.layer1, '0').conv1
    """
    layers: list[str] = [f'layer{i}.0.conv1' for i in range(1, L + 1)]
    if include_cls:
        layers: list[str] = layers + ['model.cls']
    return layers


if __name__ == '__main__':
    from types import SimpleNamespace

    # import argparse
    #
    # parser = argparse.ArgumentParser('argument for training')
    # parser.add_argument('--model', type=str, choices=['resnet12', 'resnet18', 'resnet24', 'resnet50', 'resnet101',
    #                                                   'seresnet12', 'seresnet18', 'seresnet24', 'seresnet50',
    #                                                   'seresnet101'])
    # args = parser.parse_args()
    args = SimpleNamespace(model='resnet12')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dict = {
        'resnet12': resnet12,
        'resnet18': resnet18,
        'resnet24': resnet24,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'seresnet12': seresnet12,
        'seresnet18': seresnet18,
        'seresnet24': seresnet24,
        'seresnet50': seresnet50,
        'seresnet101': seresnet101,
    }

    model = model_dict[args.model](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64).to(args.device)
    data = torch.randn(2, 3, 84, 84)
    model = model.to(args.device)
    data = data.to(args.device)
    feat, logit = model(data, is_feat=True)
    print(feat[-1].shape)
    print(logit.shape)

    print("DONE")
