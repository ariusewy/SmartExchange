import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quantize import QConv2d, QLinear


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def make_conv_layer(in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1, bias=True,
                    quantize=False, num_bits=None, num_bits_weight=None):
    if quantize:
        return QConv2d(in_channels, out_channels, kernel_size,
                       stride=stride, padding=padding, dilation=dilation,
                       groups=groups, bias=bias,
                       num_bits=num_bits, num_bits_weight=num_bits_weight)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias,)


def make_linear_layers(in_features, out_features, bias=True,
                       quantize=False, num_bits=None, num_bits_weight=None):
    if quantize:
        return QLinear(in_features, out_features, bias=bias,
                       num_bits=num_bits, num_bits_weight=num_bits_weight)
    else:
        return nn.Linear(in_features, out_features, bias=bias)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000,
                 quantize=False, num_bits=None, num_bits_weight=None):
        super(AlexNet, self).__init__()
        qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
        self.features = nn.Sequential(
            make_conv_layer(3, 64, kernel_size=11, stride=4, padding=2, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_conv_layer(64, 192, kernel_size=5, padding=2, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_conv_layer(192, 384, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            make_conv_layer(384, 256, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            make_conv_layer(256, 256, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            make_linear_layers(256 * 6 * 6, 4096, **qargs),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            make_linear_layers(4096, 4096, **qargs),
            nn.ReLU(inplace=True),
            make_linear_layers(4096, num_classes, **qargs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class QAlexNet(nn.Module):

    def __init__(self, num_classes=1000,
                 quantize=False, num_bits=None, num_bits_weight=None):
        super(QAlexNet, self).__init__()
        qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
        self.features = nn.ModuleList([
            make_conv_layer(3, 64, kernel_size=11, stride=4, padding=2, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_conv_layer(64, 192, kernel_size=5, padding=2, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            make_conv_layer(192, 384, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            make_conv_layer(384, 256, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            make_conv_layer(256, 256, kernel_size=3, padding=1, **qargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList([
            nn.Dropout(),
            make_linear_layers(256 * 6 * 6, 4096, **qargs),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            make_linear_layers(4096, 4096, **qargs),
            nn.ReLU(inplace=True),
            make_linear_layers(4096, num_classes, **qargs),
        ])

    def forward(self, x):
        qinput_infos = []
        for layer in self.features:
            if isinstance(layer, QConv2d):
                x, qinput_info = layer(x, ret_qinput=True)
                qinput_infos += [qinput_info]
                # qinput_infos += [(tuple(layer.weight.shape), qinput_info)]
            elif isinstance(layer, nn.Conv2d):
                qinput_infos += [dict(type='conv', data=x.detach(),
                                      kernel_size=layer.kernel_size[0],
                                      in_channels=layer.in_channels,
                                      out_channels=layer.out_channels,
                                      stride=layer.stride,
                                      padding=layer.padding)]
                x = layer(x)
            else:
                x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        for layer in self.classifier:
            if isinstance(layer, QLinear):
                x, qinput_info = layer(x, ret_qinput=True)
                qinput_infos += [qinput_info]
                # qinput_infos += [(tuple(layer.weight.shape), qinput_info)]
            elif isinstance(layer, nn.Linear):
                qinput_infos += [dict(type='fc', data=x.detach(),
                                      in_features=layer.in_features,
                                      out_features=layer.out_features)]
                x = layer(x)
            else:
                x = layer(x)
        return x, qinput_infos
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        # return x


def qalexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QAlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
    return model


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
    return model