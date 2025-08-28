import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
# ResNet-18 architecture sourced from: https://github.com/OxWearables/ssl-wearables
# Used for training on the UK Biobank dataset.
# ElderNet extends ResNet-18 with three additional fully connected layers, adapted for older adults using the RUSH MAP dataset.
# A regressor class is employed for fine-tuning to predict continuous gait metrics.
'''

####################################################################################################################
class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class Regressor(nn.Module):
    def __init__(self, input_size=1024, output_size=1, max_mu = 2.0, num_layers=1, batch_norm=False):
        super(Regressor, self).__init__()
        self.max_mu = max_mu
        self.bn = batch_norm
        # Create lists to hold the linear and batch norm layers
        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Add the layers
        hidden_size = input_size
        for i in range(num_layers):
            next_hidden_size = max(hidden_size // 2, 32)  # Ensure hidden size doesn't get too small
            self.linear_layers.append(nn.Linear(hidden_size, next_hidden_size))
            if self.bn:
                self.bn_layers.append(nn.BatchNorm1d(next_hidden_size))
            hidden_size = next_hidden_size

        # Output layers
        self.mu = nn.Linear(hidden_size, output_size)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def forward(self, x):
        # Pass input through all hidden layers
        if self.bn:
            for linear, bn in zip(self.linear_layers, self.bn_layers):
                x = F.relu(bn(linear(x)))
        else:
            for linear in self.linear_layers:
                x = F.relu(linear(x))

        mu = self.mu(x)
        mu = self.max_mu * torch.sigmoid(mu)  # Enforce gait speed to be between 0 and 2
        return mu


class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class LinearLayers(nn.Module):
    """
    Additional layers to tailor ElderNet for the MAP data
    """
    def __init__(self, input_size=1024, output_size=128, non_linearity=False):
        super(LinearLayers, self).__init__()
        assert input_size / 4 > 0 , "input size too small"
        assert  output_size <= (input_size/4), "output size needs to be smaller the input size/4"
        self.linear1 = torch.nn.Linear(input_size, int(input_size/2))
        self.linear2 = torch.nn.Linear(int(input_size/2), int(input_size/4))
        self.linear3 = torch.nn.Linear(int(input_size/4), output_size)
        self.relu = nn.ReLU()
        self.non_linearity = non_linearity
        weight_init(self)

    def forward(self, x):
        if self.non_linearity:
            fc1 = self.linear1(x)
            fc2 = self.linear2(self.relu(fc1))
            out = self.linear3(self.relu(fc2))
        else:
            fc1 = self.linear1(x)
            fc2 = self.linear2(fc1)
            out = self.linear3(fc2)
        return out


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        n_channels=3,
        resnet_version=1,
        output_size=1,
        epoch_len=10,
        is_classification=False,
        is_regression=False,
        max_mu=None,
        num_layers_regressor=None,
        batch_norm=False,
        feature_extractor=nn.Sequential()
    ):
        super(Resnet, self).__init__()

        # Model Configuration
        self.output_size = output_size
        epoch_len = epoch_len
        self.is_classification = is_classification
        # Regressor-related configurations
        self.is_regression = is_regression
        self.max_mu = max_mu
        self.num_layers_regressor = num_layers_regressor
        self.batch_norm = batch_norm

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        self.feature_extractor = feature_extractor
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            self.feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels


        if self.is_classification:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=self.output_size
            )

        elif self.is_regression:
            self.regressor = Regressor(input_size=out_channels, output_size=self.output_size, max_mu = self.max_mu,
                                       num_layers = self.num_layers_regressor, batch_norm= self.batch_norm
                                       )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)


    def forward(self, x):
        feats = self.feature_extractor(x)
        if self.is_classification:
            return self.classifier(feats.view(x.shape[0], -1))
        elif self.is_regression:
            return self.regressor(feats.view(x.shape[0], -1))



class ElderNet(nn.Module):
    def __init__(self, feature_extractor, non_linearity=True,
                 linear_model_input_size=1024, linear_model_output_size=50, output_size=1,
                 is_classification=False, is_regression=False, max_mu=None,
                 num_layers_regressor=None, batch_norm=False,):
        super(ElderNet, self).__init__()
        # Load the pretrained layers without classifier
        self.feature_extractor = feature_extractor
        self.output_size = output_size
        self.is_classification = is_classification #evaluating mode (fine-tuning)
        self.is_regression = is_regression #for continous labels, such as gait speed

        #Define the head of the model
        self.fc = LinearLayers(linear_model_input_size, linear_model_output_size, non_linearity)

        # Define classification layers
        if self.is_classification:
            self.classifier = Classifier(linear_model_output_size, self.output_size)

        #Define regression layers
        if self.is_regression:
            self.max_mu = max_mu
            self.num_layers_regressor =num_layers_regressor
            self.batch_norm = batch_norm

        if self.is_regression:
            self.regressor = Regressor(input_size=linear_model_output_size, output_size=self.output_size,
                                       max_mu=self.max_mu, num_layers=self.num_layers_regressor,
                                       batch_norm=self.batch_norm
                                       )

    def forward(self, x):
        features = self.feature_extractor(x)
        representation = self.fc(features.view(x.shape[0], -1))

        if self.is_classification:
            logits = self.classifier(representation)
            return logits

        elif self.is_regression:
            logits = self.regressor(representation)
            return logits


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weight_init(self, mode="fan_out", nonlinearity="relu"):
    set_seed()
    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
