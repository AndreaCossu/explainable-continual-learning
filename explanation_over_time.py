import os
import torch
import matplotlib.pyplot as plt
from utils import ReducedResNet18, IncrementalMLP, SequenceClassifier


@torch.no_grad()
def load_joint(path, relu=True):
    expl, _, _ = torch.load(os.path.join(path, 'explanations_and_examples_joint.pt'))
    # expl is a list with n_classes elements, each of which is a tensor of shape (n_test, input_shape)
    if relu:
        expl = [torch.relu(e) for e in expl]
    return expl


@torch.no_grad()
def load_final_exp(path, relu=True, exp_id=4):
    expl, _, _ = torch.load(os.path.join(path, f'explanations_and_examples_{exp_id}.pt'))
    if relu:
        expl = [torch.relu(e) for e in expl]
    return expl


@torch.no_grad()
def znorm(x):
    return (x - x.mean()) / x.std()


@torch.no_grad()
def compare_joint_vs_cl_pool(expl_joint, expl_cl, examples_per_class=50, pool_size=4, pool2d=True, use_znorm=False):
    pool = torch.nn.AvgPool2d(pool_size) if pool2d else torch.nn.AvgPool1d(pool_size)
    mse_0_list, mse_1_list = [], []
    for class_id, (ej, ec) in enumerate(zip(expl_joint, expl_cl)):
        if not pool2d:
            # sequence-length last
            ej = ej.transpose(1, 2)
            ec = ec.transpose(1, 2)

        if use_znorm:
            class_0_mse = torch.nn.functional.mse_loss(pool(znorm(ej[:examples_per_class])), pool(znorm(ec[:examples_per_class])))
            class_1_mse = torch.nn.functional.mse_loss(pool(znorm(ej[examples_per_class:])), pool(znorm(ec[examples_per_class:])))
        else:
            class_0_mse = torch.nn.functional.mse_loss(pool(ej[:examples_per_class]), pool(ec[:examples_per_class]))
            class_1_mse = torch.nn.functional.mse_loss(pool(ej[examples_per_class:]), pool(ec[examples_per_class:]))
        mse_0_list.append(class_0_mse)
        mse_1_list.append(class_1_mse)

    return mse_0_list, mse_1_list


@torch.no_grad()
def compare_joint_vs_cl_sum(expl_joint, expl_cl, examples_per_class=50):
    mse_0_list, mse_1_list = [], []
    for class_id, (ej, ec) in enumerate(zip(expl_joint, expl_cl)):
        class_0_mse = (ej[:examples_per_class].sum() - ec[:examples_per_class].sum())**2 / float(ej[:examples_per_class].numel())
        class_1_mse = (ej[examples_per_class:].sum() - ec[examples_per_class:].sum())**2 / float(ej[examples_per_class:].numel())
        mse_0_list.append(class_0_mse)
        mse_1_list.append(class_1_mse)

    return mse_0_list, mse_1_list


@torch.no_grad()
def compare_joint_vs_cl_elementwise(expl_joint, expl_cl, examples_per_class=50):
    mse_0_list, mse_1_list = [], []
    for class_id, (ej, ec) in enumerate(zip(expl_joint, expl_cl)):
        class_0_mse = torch.sum((ej[:examples_per_class] - ec[:examples_per_class])**2) / float(ej[:examples_per_class].numel())
        class_1_mse = torch.sum((ej[examples_per_class:] - ec[examples_per_class:])**2) / float(ej[examples_per_class:].numel())
        mse_0_list.append(class_0_mse)
        mse_1_list.append(class_1_mse)

    return mse_0_list, mse_1_list


@torch.no_grad()
def plot_expl_difference(mse_0_list, mse_1_list, path, suffix=''):
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.plot(list(range(10)), mse_0_list, 'r-', label='target class 0')
    plt.plot(list(range(10)), mse_1_list, 'g--', label='target class 1')
    plt.scatter(0, mse_0_list[0], s=70, c='r')
    plt.scatter(1, mse_1_list[1], s=70, c='g')
    plt.legend(loc='upper right', fontsize=28)
    plt.xlim(-0.2, 9)
    plt.xticks(list(range(10)), [str(e) for e in list(range(10))])
    plt.savefig(os.path.join(path, f'explanation_difference_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    base_path = '/home/a.cossu/explainable-continual-learning/results'

    strategies = ['naive', 'replay', 'gss', 'joint']
    datasets = ['mnist', 'cifar', 'speech']
    dat_to_model = {'mnist': ['mlp'], 'cifar': ['cnn'], 'speech': ['rnn', 'esn', 'ron', 'cnn']}
    explanators = ['shap', 'lift']
    max_pools = {'mnist': 28, 'cifar': 32, 'speech': 10}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for use_relu in [True, False]:
        suffix = '' if use_relu else '_posneg'
        for strategy in strategies:
            for dataset in datasets:
                for explanator in explanators:
                    models = dat_to_model[dataset]
                    for model in models:
                        strategy_path = os.path.join(base_path, f'{strategy}_{dataset}_{model}_{explanator}')
                        joint_path = os.path.join(base_path, f'joint_{dataset}_{model}_{explanator}')

                        if strategy == 'joint':
                            continue

                        expl_joint = load_joint(joint_path, relu=use_relu)
                        expl_cl = load_final_exp(strategy_path, relu=use_relu)
                        mse0, mse1 = compare_joint_vs_cl_pool(expl_joint,
                                                              expl_cl,
                                                              examples_per_class=50, pool_size=4,
                                                              use_znorm=True)
                        plot_expl_difference(mse0, mse1, strategy_path, suffix='pool_4_norm' + suffix)
                        mse0, mse1 = compare_joint_vs_cl_pool(expl_joint, expl_cl,
                                                              examples_per_class=50,
                                                              pool_size=max_pools[dataset],
                                                              pool2d=(dataset != 'speech'),
                                                              use_znorm=True)
                        plot_expl_difference(mse0, mse1, strategy_path, suffix='pool_max' + suffix)
                        mse0, mse1 = compare_joint_vs_cl_sum(expl_joint, expl_cl, examples_per_class=50)
                        plot_expl_difference(mse0, mse1, strategy_path, suffix='sum' + suffix)
                        mse0, mse1 = compare_joint_vs_cl_elementwise(expl_joint, expl_cl, examples_per_class=50)
                        plot_expl_difference(mse0, mse1, strategy_path, suffix='elementwise' + suffix)