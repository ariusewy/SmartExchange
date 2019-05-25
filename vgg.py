import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from quantize import QConv2d, QLinear


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


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


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True,
                 quantize=False, num_bits=None, num_bits_weight=None):
        super(VGG, self).__init__()
        qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            make_linear_layers(512 * 7 * 7, 4096, **qargs),
            nn.ReLU(True),
            nn.Dropout(),
            make_linear_layers(4096, 4096, **qargs),
            nn.ReLU(True),
            nn.Dropout(),
            make_linear_layers(4096, num_classes, **qargs),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class QVGG(nn.Module):

    def __init__(self, cfg, num_classes=1000, init_weights=True,
                 quantize=True, num_bits=8, num_bits_weight=8):
        super(QVGG, self).__init__()
        qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
        feature_layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = make_conv_layer(in_channels, v, kernel_size=3,
                                         padding=1, **qargs)
                feature_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.ModuleList(feature_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.ModuleList([
            make_linear_layers(512 * 7 * 7, 4096, **qargs),
            nn.ReLU(True),
            nn.Dropout(),
            make_linear_layers(4096, 4096, **qargs),
            nn.ReLU(True),
            nn.Dropout(),
            make_linear_layers(4096, num_classes, **qargs),
        ])
        if init_weights:
            self._initialize_weights()

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
                                      stride=layer.stride, padding=layer.padding)]
                x = layer(x)
            else:
                x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, **qargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = make_conv_layer(in_channels, v, kernel_size=3, padding=1, **qargs)
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    quantize = kwargs.pop('quantize', False)
    num_bits = kwargs.pop('num_bits', None)
    num_bits_weight = kwargs.pop('num_bits', None)
    qargs = dict(quantize=quantize, num_bits=num_bits,
                 num_bits_weight=num_bits_weight)
    kwargs.update(**qargs)
    model = VGG(make_layers(cfg['D'], **qargs), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


def qvgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    quantize = kwargs.pop('quantize', True)
    num_bits = kwargs.pop('num_bits', 8)
    num_bits_weight = kwargs.pop('num_bits', 8)
    qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
    kwargs.update(**qargs)
    model = QVGG(cfg['D'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    quantize = kwargs.pop('quantize', False)
    num_bits = kwargs.pop('num_bits', None)
    num_bits_weight = kwargs.pop('num_bits', None)
    qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
    kwargs.update(**qargs)
    model = VGG(make_layers(cfg['E'], **qargs), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    quantize = kwargs.pop('quantize', False)
    num_bits = kwargs.pop('num_bits', None)
    num_bits_weight = kwargs.pop('num_bits', None)
    qargs = dict(quantize=quantize, num_bits=num_bits, num_bits_weight=num_bits_weight)
    kwargs.update(**qargs)
    model = VGG(make_layers(cfg['E'], batch_norm=True, **qargs), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
    return model