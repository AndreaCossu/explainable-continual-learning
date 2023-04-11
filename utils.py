import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import shap
from torch.utils.data import DataLoader
from torch import nn, relu
from torch.nn.functional import avg_pool2d
from avalanche.models import IncrementalClassifier


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut1 = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                                      stride=stride, bias=True)
            self.shortcut2 = nn.BatchNorm2d(self.expansion * planes)
            self.shortcut = True
        else:
            self.shortcut = False

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            tmp = self.shortcut1(x)
            out += self.shortcut2(tmp)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        return out

class ReducedResNet18(nn.Module):
    """
    As from GEM paper, a smaller version of ResNet18, with three times less feature maps across all layers.
    It employs multi-head output layer.
    """
    def __init__(self, size_before_classifier=160):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        self.classifier = IncrementalClassifier(in_features=size_before_classifier, initial_out_features=2)
    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class IncrementalMLP(nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = IncrementalClassifier(in_features=hidden_size, initial_out_features=2)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x



def get_shap_background_and_test(dataset, num_background=700, num_inputs_per_class=3, classes=[0, 1]):
    batch_size = num_background + num_inputs_per_class*len(classes)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    xs, ys = [], []
    for batch in test_dataloader:
        x, y, _ = batch
        xs.append(x)
        ys.append(y)
        tryagain = False
        for counter in [(y == c).sum() for c in classes]:
            if counter.item() < num_inputs_per_class:  # class has not enough samples in current batch
                tryagain = True
        if not tryagain:
            break
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    test_inputs, background_inputs = [], []
    for c in classes:
        idx = torch.nonzero(ys == c)
        idx_test = idx[:num_inputs_per_class]
        idx_background = idx[num_inputs_per_class:]
        test_inputs.append(xs[idx_test].squeeze(1))
        background_inputs.append(xs[idx_background].squeeze(1))
        print(f"For class {c} using {test_inputs[-1].shape[0]} "
              f"examples for test and {background_inputs[-1].shape[0]} examples for background")
    test_inputs = torch.cat(test_inputs, dim=0)
    background_inputs = torch.cat(background_inputs, dim=0)

    return background_inputs, test_inputs


def compute_shap(model, shap_background, shap_test, device):
    e = shap.DeepExplainer(model, shap_background.to(device))
    shap_values = e.shap_values(shap_test.to(device))
    return shap_values, shap_test


def plot_shap(shap_values, test_inputs, save_path, name='test'):
    # for each channel, print shap
    for c in range(test_inputs.shape[1]):
        shap_numpy = [np.swapaxes(np.swapaxes(s[:, c, ...][:, np.newaxis, ...], 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_inputs.numpy()[:, c, ...][:, np.newaxis, ...], 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(os.path.join(save_path, f'{name}_c{c}.png'))
