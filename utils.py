import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import shap
import captum
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, relu
from torch.nn.functional import avg_pool2d
from avalanche.models import IncrementalClassifier
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
from matplotlib.colors import LinearSegmentedColormap
from avalanche.training.plugins import GSS_greedyPlugin


class MyGSS(GSS_greedyPlugin):
    def after_forward(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        After every forward this function select sample to fill
        the memory buffer based on cosine similarity
        """
        if not strategy.rnn:
            strategy.model.eval()

        # Compute the gradient dimension
        grad_dims = []
        for param in strategy.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = (
            self.ext_mem_list_x.size(0) - self.ext_mem_list_current_index
        )
        if place_left <= 0:  # buffer full

            batch_sim, mem_grads = self.get_batch_sim(
                strategy,
                grad_dims,
                batch_x=strategy.mb_x,
                batch_y=strategy.mb_y,
            )

            if batch_sim < 0:
                buffer_score = self.buffer_score[
                    : self.ext_mem_list_current_index
                ].cpu()

                buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                    (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                )

                # draw candidates for replacement from the buffer
                index = torch.multinomial(
                    buffer_sim, strategy.mb_x.size(0), replacement=False
                ).to(strategy.device)

                # estimate the similarity of each sample in the received batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, strategy.mb_x, strategy.mb_y
                )

                # normalize to [0,1]
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = (
                    (self.buffer_score[index] + 1) / 2
                ).unsqueeze(1)
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(
                    torch.cat(
                        (scaled_batch_item_sim, buffer_repl_batch_sim), dim=1
                    ),
                    1,
                    replacement=False,
                )
                # replace samples with outcome =1
                added_indx = torch.arange(
                    end=batch_item_sim.size(0), device=strategy.device
                )
                sub_index = outcome.squeeze(1).bool()
                self.ext_mem_list_x[index[sub_index]] = strategy.mb_x[
                    added_indx[sub_index]
                ].clone()
                self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[
                    added_indx[sub_index]
                ].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[
                    added_indx[sub_index]
                ].clone()
        else:
            offset = min(place_left, strategy.mb_x.size(0))
            updated_mb_x = strategy.mb_x[:offset]
            updated_mb_y = strategy.mb_y[:offset]

            # first buffer insertion
            if self.ext_mem_list_current_index == 0:
                batch_sample_memory_cos = (
                    torch.zeros(updated_mb_x.size(0)) + 0.1
                )
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(
                    strategy=strategy,
                    grad_dims=grad_dims,
                    gss_batch_size=len(strategy.mb_x),
                )

                # estimate a score for each added sample
                batch_sample_memory_cos = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, updated_mb_x, updated_mb_y
                )

            curr_idx = self.ext_mem_list_current_index
            self.ext_mem_list_x[curr_idx : curr_idx + offset].data.copy_(
                updated_mb_x
            )
            self.ext_mem_list_y[curr_idx : curr_idx + offset].data.copy_(
                updated_mb_y
            )
            self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                batch_sample_memory_cos
            )
            self.ext_mem_list_current_index += offset

        strategy.model.train()

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
    def __init__(self, size_before_classifier=160, initial_out_features=2):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        self.classifier = IncrementalClassifier(in_features=size_before_classifier, initial_out_features=initial_out_features)
    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class CNN1D(nn.Module):
    def __init__(self, initial_out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            IncrementalClassifier(1920, initial_out_features=initial_out_features)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        return out


class IncrementalMLP(nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
        initial_out_features=2
    ):
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
        self.classifier = IncrementalClassifier(in_features=hidden_size, initial_out_features=initial_out_features)
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


class SequenceClassifier(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, rnn_layers=1, batch_first=True, initial_out_features=2
    ):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=batch_first,
        )
        self.classifier = IncrementalClassifier(in_features=hidden_size, initial_out_features=initial_out_features)


    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1] if self.batch_first else out[-1]
        out = self.classifier(out)
        return out


def create_speech_benchmark(root, num_words=10, test_split=0.2):
    assert num_words % 2 == 0, "The Speech dataset must have an even number of classes."
    classfiles = list(sorted(os.listdir(root)))[:num_words]
    all_tensors = [torch.from_numpy(np.load(os.path.join(root, classfile))).float() for classfile in classfiles]

    tensors = []
    for i in range(len(all_tensors)):
        train_len = int(all_tensors[i].shape[0] * (1-test_split))
        train_tensor = all_tensors[i][:train_len]
        mean = train_tensor.view(-1, train_tensor.shape[-1]).mean(dim=0)
        std = train_tensor.view(-1, train_tensor.shape[-1]).std(dim=0)
        train_tensor = (train_tensor - mean) / std
        test_tensor = all_tensors[i][train_len:]
        test_tensor = (test_tensor - mean) / std
        tensors.append({'train': train_tensor, 'test': test_tensor})

    datasets = {'train': [], 'test': []}
    for mode in ['train', 'test']:
        for i in range(0, num_words, 2):
            classes_1 = (torch.ones(tensors[i][mode].shape[0]) * i).long()
            classes_2 = (torch.ones(tensors[i+1][mode].shape[0]) * (i+1)).long()
            targets = torch.cat((classes_1, classes_2), dim=0)
            dataset = TensorDataset(
                torch.cat((tensors[i][mode], tensors[i+1][mode]), dim=0),
                targets
            )
            datasets[mode].append(make_classification_dataset(dataset, targets=targets))
    benchmark = dataset_benchmark(train_datasets=datasets['train'], test_datasets=datasets['test'])

    return benchmark

def get_background_and_test(dataset, num_background=700, num_inputs_per_class=3, classes=(0, 1),
                                 collate_fn=None, split_equal=False):

    batch_size = num_background + num_inputs_per_class*len(classes)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn)

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

    test_inputs, background_inputs, test_targets = [], [], []
    for c in classes:
        idx = torch.nonzero(ys == c)
        idx_test = idx[:num_inputs_per_class]
        if split_equal:
            idx_background = idx[num_inputs_per_class:num_inputs_per_class*2]
        else:
            idx_background = idx[num_inputs_per_class:]
        test_inputs.append(xs[idx_test].squeeze(1))
        test_targets.append(ys[idx_test].squeeze(1))
        background_inputs.append(xs[idx_background].squeeze(1))
        print(f"For class {c} using {test_inputs[-1].shape[0]} "
              f"examples for test and {background_inputs[-1].shape[0]} examples for background")
    test_inputs = torch.cat(test_inputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    background_inputs = torch.cat(background_inputs, dim=0)

    return background_inputs, test_inputs, test_targets


def plot_explanations_grid(shap_values, test_inputs, save_path, name='test', num_plots=6):
    per_class_els = int(num_plots / 2)
    test_inputs = torch.cat((test_inputs[:per_class_els], test_inputs[-per_class_els:]))
    shap_values = [torch.cat((s[:per_class_els], s[-per_class_els:])) for s in shap_values]
    if len(shap_values[0].shape) != 4:
        # plot spectrogram as image
        shap_numpy = [s.detach().numpy()[..., np.newaxis] for s in shap_values]
        test_numpy = test_inputs.detach().numpy()[..., np.newaxis]
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(os.path.join(save_path, f'{name}.png'))

        # plot time series averaged over features and weighted by shap
        shap_numpy = [s.squeeze(-1).mean(axis=-1) for s in shap_numpy]
        test_numpy = test_inputs.detach().numpy().mean(axis=-1)
        fig, ax = plt.subplots(test_numpy.shape[0], len(shap_numpy)+1, figsize=(20, 5))
        for i in range(test_numpy.shape[0]):
            ax[i, 0].plot(test_numpy[i], 'g-')
            ax[i, 0].set_ylim(-1, 3)
        default_cmap = plt.get_cmap('Greys')
        for i in range(test_numpy.shape[0]):
            for j in range(len(shap_numpy)):
                shap_numpy_norm = (shap_numpy[j][i] - shap_numpy[j][i].min()) / (shap_numpy[j][i].max() - shap_numpy[j][i].min())
                ax[i, j+1].plot(np.arange(test_numpy.shape[1]), test_numpy[i], c='white')
                for k in range(test_numpy[i].shape[0]):
                    ax[i, j+1].plot([k], test_numpy[i][k],
                                  linewidth=0., marker='o',
                                  c=default_cmap(shap_numpy_norm[k]),
                                  markersize=0.5)
                ax[i, j+1].set_ylim(-1, 3)
        plt.savefig(os.path.join(save_path, f'{name}_timeseries.png'))
    else:
        # plot average over channels
        if len(shap_values[0].squeeze().shape) == 4:
            shap_numpy = [np.swapaxes(np.swapaxes(s.sum(axis=1).detach().numpy()[:, np.newaxis, ...], 1, -1), 1, 2) for s in shap_values]
            shap_numpy = [torch.relu(torch.from_numpy(s)) for s in shap_numpy]
            test_numpy = np.swapaxes(np.swapaxes(test_inputs.detach().numpy().sum(axis=1)[:, np.newaxis, ...], 1, -1), 1, 2)
            shap.image_plot(shap_numpy, -test_numpy)
            plt.savefig(os.path.join(save_path, f'{name}_sum_channels.png'))

        # for each channel, print shap
        for c in range(test_inputs.shape[1]):
            shap_numpy = [np.swapaxes(np.swapaxes(s.detach().numpy()[:, c, ...][:, np.newaxis, ...], 1, -1), 1, 2) for s in shap_values]
            test_numpy = np.swapaxes(np.swapaxes(test_inputs.detach().numpy()[:, c, ...][:, np.newaxis, ...], 1, -1), 1, 2)
            shap_numpy = [torch.relu(torch.from_numpy(s)) for s in shap_numpy]
            shap.image_plot(shap_numpy, -test_numpy)
            plt.savefig(os.path.join(save_path, f'{name}_c{c}.png'))


def plot_shap_single(explanations, shap_test, folder_path, name='single'):
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)
    for ex_id in range(explanations.shape[0]):
        captum.attr.visualization.visualize_image_attr_multiple(
            np.transpose(explanations[ex_id].cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(shap_test[ex_id].cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "absolute_value"],
            cmap=default_cmap,
            show_colorbar=True,
        )
        plt.savefig(os.path.join(folder_path, f"{name}_ex{ex_id}.png"))
