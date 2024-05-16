"""Module for building resnet module according to different config file."""

from typing import Any, Iterable, List, Optional, Sequence, Type, Union

from darwinai.torch.builder import BlockSpec
from torch import Tensor, nn
from torch.nn import Module


def activation() -> Module:
    """Helper for building an activation layer."""
    return nn.ReLU(inplace=True)


def conv2d(
    w_in: int,
    w_out: int,
    k: int,
    *,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False
) -> Module:
    """Helper for building a conv2d layer."""
    # assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in: int) -> Module:
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=1e-5, momentum=0.1)


def pool2d(_w_in: int, k: int, *, stride: int = 1) -> Module:
    """Helper for building a pool2d layer."""
    # assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


class BasicTransform(Module):
    """Basic transformation: 3x3, BN, AF, 3x3, BN."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(BasicTransform, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.stride = stride
        self.inplanes = inplanes
        self.channels = channels
        self.conv1 = conv2d(self.inplanes, self.channels, 3, stride=self.stride)
        self.bn1 = norm2d(self.channels)
        self.relu = activation()
        self.conv2 = conv2d(self.channels, self.channels, 3)
        self.bn2 = norm2d(self.channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, BN, AF, 3x3, BN, AF, 1x1, BN."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(BottleneckTransform, self).__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.conv1 = conv2d(inplanes, width, 1)
        self.bn1 = norm2d(width)
        self.conv2 = conv2d(
            width, width, 3, stride=stride, groups=groups, dilation=dilation
        )
        self.bn2 = norm2d(width)
        self.conv3 = conv2d(width, channels * self.expansion, 1)
        self.bn3 = norm2d(channels * self.expansion)
        self.relu = activation()
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.inplanes = inplanes
        self.channels = channels
        self.groups = groups
        self.dilation = dilation
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def get_transformation_function(
    func_name: str,
) -> Union[Type[BottleneckTransform], Type[BasicTransform]]:
    """Returns the transformation function for ResNet Module."""
    if func_name == "basic":
        return BasicTransform
    elif func_name == "bottleneck":
        return BottleneckTransform
    else:
        raise (ValueError("Function not available"))


class ResHead(Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in: int, num_classes: int) -> None:
        super(ResHead, self).__init__()
        self.w_in = w_in
        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, num_classes, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResStemIN(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in: int, w_out: int) -> None:
        super(ResStemIN, self).__init__()
        self.w_in = w_in
        self.w_out = w_out
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.children():
            x = layer(x)
        return x


class Layer(Module):
    """Class for each layer of the resnet model which accepts parameterized channel
    and depth."""

    def __init__(
        self,
        channels: int,
        depth: int,
    ) -> None:
        """Initialize layer attributes such as channels and depth."""
        super(Layer, self).__init__()
        self.base_width = 64
        self.groups = 1
        self.channels = channels
        self.depth = depth

    def _make_layer(
        self,
        block_type: Union[Type[BottleneckTransform], Type[BasicTransform]],
        inplanes: int,
        dilation: int,
        stride: int,
        dilate: bool = False,
    ) -> Module:
        """This function is responsible for making a layer based on the parameterized
        channels and depth of that layer."""
        downsample = None
        self.dilation = dilation
        self.inplanes = inplanes
        previous_dilation = self.dilation
        channels = self.channels
        depth = self.depth
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != channels * block_type.expansion:
            downsample = nn.Sequential(
                conv2d(
                    self.inplanes, channels * block_type.expansion, 1, stride=stride
                ),
                norm2d(channels * block_type.expansion),
            )
        layers = []
        layers.append(
            block_type(
                self.inplanes,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = channels * block_type.expansion
        for _ in range(1, depth):
            layers.append(
                block_type(
                    self.inplanes,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)


class ResNet(Module):
    """ResNet model."""

    def __init__(
        self,
        blockspecs: Optional[List[BlockSpec]],
        num_classes: int,
    ) -> None:
        super(ResNet, self).__init__()
        self.blockspecs = []  # type: List[BlockSpec]
        if blockspecs is None:
            self.blockspecs = [
                BlockSpec(64, 3, True, False),
                BlockSpec(128, 4, True, False),
                BlockSpec(256, 6, True, False),
                BlockSpec(512, 3, True, False),
            ]
        else:
            self.blockspecs = blockspecs

        self.blocks = [Layer(b.channels, b.depth) for b in self.blockspecs]

        self.replace_stride_with_dilation = [False, False, False, False]
        self.block_types = ["bottleneck", "bottleneck", "bottleneck", "bottleneck"]

        self.num_channels = len(self.block_types)
        self.blocks_list = []  # type: Any
        self.num_classes = num_classes
        self.construct_model()

    def construct_model(self) -> None:
        """This function constructs the model by iterating over the list of modules
        specified by the user."""
        self.inplanes = 64
        self.stem = ResStemIN(3, self.inplanes)

        for i, block in enumerate(self.blocks):
            if i == 0:
                self.blocks_list.append(
                    block._make_layer(
                        get_transformation_function(self.block_types[i]),
                        self.inplanes,
                        dilation=1,
                        stride=1,
                    )
                )
            else:
                self.blocks_list.append(
                    block._make_layer(
                        get_transformation_function(self.block_types[i]),
                        self.blocks[i - 1].inplanes,
                        self.blocks[i - 1].dilation,
                        stride=2,
                        dilate=self.replace_stride_with_dilation[i - 1],
                    )
                )
        self.blocks_list = nn.ModuleList(self.blocks_list)
        self.head = ResHead(
            self.blockspecs[self.num_channels - 1].channels
            * get_transformation_function(
                self.block_types[self.num_channels - 1]
            ).expansion,
            self.num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for i in range(self.num_channels):
            x = self.blocks_list[i](x)
        x = self.head(x)
        return x
