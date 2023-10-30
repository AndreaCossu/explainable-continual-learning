import argparse
import torch
import os
import torchvision.transforms as transforms
import pickle
from torch.optim import SGD, Adam
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, Replay, JointTraining
from utils import plot_explanations_grid, IncrementalMLP, get_background_and_test, \
    ReducedResNet18, SequenceClassifier, create_speech_benchmark, MyGSS, CNN1D
import json
from captum.attr import GradientShap, DeepLift
from esn import DeepReservoirClassifier
from ron import RON


def main(args):
    strategies = ['naive', 'gss', 'replay', 'joint']
    exp_names = [f"{s}_{args['dataset']}_{args['model']}" for s in strategies]
    folder_paths = dict(
        [(e.split('_')[0], os.path.join(args['experiment_folder'], e)) for e in exp_names]
    )
    for k, f in folder_paths.items():
        os.makedirs(f, exist_ok=True)
        with open(os.path.join(f, 'args.json'), 'w') as f:
            json.dump(args, f)

    device = torch.device(
        f"cuda:{args['cuda']}" if torch.cuda.is_available() and args['cuda'] >= 0 else "cpu"
    )

    strategy_objects = {}
    models = {}
    optimizers = {}

    if args['dataset'] == 'cifar':
        assert args['model'] == 'cnn'
        input_size = [3, 32, 32]
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        benchmark = SplitCIFAR10(
            5,
            return_task_id=False,
            fixed_class_order=list(range(10)),
            train_transform=train_transform,
            eval_transform=eval_transform,
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
        models['naive'] = ReducedResNet18()
        models['gss'] = ReducedResNet18()
        models['replay'] = ReducedResNet18()
        models['joint'] = ReducedResNet18(initial_out_features=10)

        optimizers['naive'] = Adam(models['naive'].parameters(), lr=args['lr'])
        optimizers['gss'] = Adam(models['gss'].parameters(), lr=args['lr'])
        optimizers['replay'] = Adam(models['replay'].parameters(), lr=args['lr'])
        optimizers['joint'] = Adam(models['joint'].parameters(), lr=args['joint_lr'])
    elif args['dataset'] == 'mnist':
        assert args['model'] == 'mlp'
        input_size = [1, 28, 28]
        benchmark = SplitMNIST(5, return_task_id=False,
                               fixed_class_order=list(range(10)),
                               shuffle=True,
                               class_ids_from_zero_in_each_exp=False)
        models['naive'] = IncrementalMLP(num_classes=10, input_size=28*28, hidden_size=256, hidden_layers=1)
        models['gss'] = IncrementalMLP(num_classes=10, input_size=28*28, hidden_size=256, hidden_layers=1)
        models['replay'] = IncrementalMLP(num_classes=10, input_size=28*28, hidden_size=256, hidden_layers=1)
        models['joint'] = IncrementalMLP(num_classes=10, input_size=28*28, hidden_size=256, hidden_layers=1,
                                         initial_out_features=10)

        optimizers['naive'] = SGD(models['naive'].parameters(), lr=args['lr'])
        optimizers['gss'] = SGD(models['gss'].parameters(), lr=args['lr'])
        optimizers['replay'] = SGD(models['replay'].parameters(), lr=args['lr'])
        optimizers['joint'] = SGD(models['joint'].parameters(), lr=args['joint_lr'])
    elif args['dataset'] == 'speech':
        assert args['model'] in ['esn', 'rnn', 'ron']

        input_size = [101, 40]
        benchmark = create_speech_benchmark(args['speech_root'], num_words=10, test_split=0.2)
        if args['model'] == 'esn':
            models['naive'] = DeepReservoirClassifier(40, units=args['hidden_size'], layers=1, spectral_radius=args['spectral_radius'],
                                                      input_scaling=1, feedforward_layers=1, connectivity_input=100,
                                                      connectivity_recurrent=50, leaky=0.1)
            models['gss'] = DeepReservoirClassifier(40, units=args['hidden_size'], layers=1, spectral_radius=args['spectral_radius'],
                                                      input_scaling=1, feedforward_layers=1, connectivity_input=100,
                                                      connectivity_recurrent=50, leaky=0.1)
            models['replay'] = DeepReservoirClassifier(40, units=args['hidden_size'], layers=1, spectral_radius=args['spectral_radius'],
                                                      input_scaling=1, feedforward_layers=1, connectivity_input=100,
                                                      connectivity_recurrent=50, leaky=0.1)
            models['joint'] = DeepReservoirClassifier(40, units=args['hidden_size'], layers=1, spectral_radius=args['spectral_radius'],
                                                      input_scaling=1, feedforward_layers=1, connectivity_input=100,
                                                      connectivity_recurrent=50, leaky=0.1, initial_out_features=10)
        elif args['model'] == 'cnn':
            models['naive'] = CNN1D()
            models['gss'] = CNN1D()
            models['replay'] = CNN1D()
            models['joint'] = CNN1D(initial_out_features=10)
        elif args['model'] == 'ron':
            models['naive'] = RON(n_inp=40, n_hid=args['hidden_size'], dt=args['dt'],
                                  gamma_min=args['gamma_min'], gamma_max=args['gamma_max'],
                                  eps_min=args['epsilon_min'], eps_max=args['epsilon_max'],
                                  rho=args['spectral_radius'], input_scaling=args['input_scaling'],
                                  device=device, return_sequences=False)
            models['gss'] = RON(n_inp=40, n_hid=args['hidden_size'], dt=args['dt'],
                                  gamma_min=args['gamma_min'], gamma_max=args['gamma_max'],
                                  eps_min=args['epsilon_min'], eps_max=args['epsilon_max'],
                                  rho=args['spectral_radius'], input_scaling=args['input_scaling'],
                                  device=device, return_sequences=False)
            models['replay'] = RON(n_inp=40, n_hid=args['hidden_size'], dt=args['dt'],
                                  gamma_min=args['gamma_min'], gamma_max=args['gamma_max'],
                                  eps_min=args['epsilon_min'], eps_max=args['epsilon_max'],
                                  rho=args['spectral_radius'], input_scaling=args['input_scaling'],
                                  device=device, return_sequences=False)
            models['joint'] = RON(n_inp=40, n_hid=args['hidden_size'], dt=args['dt'],
                                  gamma_min=args['gamma_min'], gamma_max=args['gamma_max'],
                                  eps_min=args['epsilon_min'], eps_max=args['epsilon_max'],
                                  rho=args['spectral_radius'], input_scaling=args['input_scaling'],
                                  device=device, return_sequences=False, initial_out_features=10)
        elif args['model'] == 'rnn':
            models['naive'] = SequenceClassifier(input_size=40, hidden_size=256, rnn_layers=2, batch_first=True)
            models['gss'] = SequenceClassifier(input_size=40, hidden_size=256, rnn_layers=2, batch_first=True)
            models['replay'] = SequenceClassifier(input_size=40, hidden_size=256, rnn_layers=2, batch_first=True)
            models['joint'] = SequenceClassifier(input_size=40, hidden_size=256, rnn_layers=2, batch_first=True,
                                                 initial_out_features=10)
        else:
            raise ValueError("Wrong model name")

        optimizers['naive'] = Adam(models['naive'].parameters(), lr=args['lr'])
        optimizers['gss'] = Adam(models['gss'].parameters(), lr=args['lr'])
        optimizers['replay'] = Adam(models['replay'].parameters(), lr=args['lr'])
        optimizers['joint'] = Adam(models['joint'].parameters(), lr=args['joint_lr'])
    else:
        raise ValueError("Unrecognized dataset name")

    # <-------------------------- GET BACKGROUND EXAMPLES
    background, back_test, back_test_targets = get_background_and_test(
        dataset=benchmark.test_stream[0].dataset,
        num_background=args['num_background'],
        num_inputs_per_class=args['num_test_per_class'],
        classes=(0, 1))

    # <----------------------------- START JOINT
    if args['explanator'] == 'shap':
        joint_explanator = GradientShap(models['joint'])
    elif args['explanator'] == 'lift':
        joint_explanator = DeepLift(models['joint'])
    else:
        raise ValueError("Unrecognized explanator name")

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]
    training_metrics = []
    evaluation_metrics = [
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
    ]
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )
    results_joint = []
    joint_strategy = JointTraining(
        model=models['joint'],
        optimizer=optimizers['joint'],
        criterion=torch.nn.CrossEntropyLoss(),
        train_epochs=args['joint_epochs'],
        evaluator=evaluator,
        device=device,
        train_mb_size=args['joint_train_mb_size'],
        eval_mb_size=64
    )
    joint_strategy.train(benchmark.train_stream)
    results_joint.append(joint_strategy.eval(benchmark.test_stream))
    explanations = []
    for c in range(10):
        expl = joint_explanator.attribute(back_test.to(device),
                                    background.to(device),
                                    target=c).cpu()

        explanations.append(expl)
    plot_explanations_grid(explanations, back_test,
                   folder_paths['joint'], name=f'joint_{args["dataset"]}', num_plots=args['num_plots'])
    torch.save([explanations, back_test, back_test_targets], os.path.join(folder_paths['joint'], f'explanations_and_examples_joint.pt'))
    torch.save(models['joint'].state_dict(), os.path.join(folder_paths['joint'], f'joint_model.pt'))
    with open(os.path.join(folder_paths['joint'], 'metrics_joint.pickle'), 'wb') as f:
        pickle.dump(results_joint, f)

    # <------------ START CL STRATEGIES
    cl_strategies = {}
    for s in ['naive', 'gss', 'replay']:
        if s == 'naive':
            cl_strategies['naive'] = Naive(
                model=models['naive'],
                optimizer=optimizers['naive'],
                train_epochs=args['epochs'],
                evaluator=evaluator,
                device=device,
                train_mb_size=args['train_mb_size'],
                eval_mb_size=64,
            )
        elif s == 'replay':
            cl_strategies['replay'] = Replay(
                model=models['replay'],
                optimizer=optimizers['replay'],
                criterion=torch.nn.CrossEntropyLoss(),
                train_epochs=args['epochs'],
                evaluator=evaluator,
                device=device,
                train_mb_size=args['train_mb_size'],
                eval_mb_size=64,
                mem_size=args['replay_mem_size'],
            )
        elif s == 'gss':
            cl_strategies['gss'] = Naive(
                model=models['gss'],
                optimizer=optimizers['gss'],
                criterion=torch.nn.CrossEntropyLoss(),
                plugins=[MyGSS(input_size=input_size, mem_size=args['replay_mem_size'], mem_strength=1)],
                train_epochs=args['epochs'],
                evaluator=evaluator,
                device=device,
                train_mb_size=args['train_mb_size'],
                eval_mb_size=64
            )
            cl_strategies['gss'].rnn = True if args['dataset'] == 'speech' else False
        strategy_objects[s] = {'strategy': cl_strategies[s],
                               'model': models[s]}

    for strategy_name, v in strategy_objects.items():
        results = []
        cl_strategy = v['strategy']
        model = v['model']
        if args['explanator'] == 'shap':
            explanator = GradientShap(model)
        elif args['explanator'] == 'lift':
            explanator = DeepLift(model)
        else:
            raise ValueError("Unrecognized explanator name")

        for i, experience in enumerate(benchmark.train_stream):
            cl_strategy.train(
                experience,
                num_workers=0,
                drop_last=True,
            )
            results.append(cl_strategy.eval(benchmark.test_stream))

            classes_so_far = set()
            for e in range(i+1):
                classes_so_far = classes_so_far.union(set(benchmark.classes_in_experience['train'][e]))
            classes_so_far = list(classes_so_far)

            # predict explanations values related to all the classes so far.
            # explanations is a list with n_classes elements. Each element is a numpy array of shape (num_test, input_shape),
            # where input_shape is (1, 28, 28) for MNIST, (3, 32, 32) for CIFAR10, (101, 40) for Speech.
            explanations = []
            for c in classes_so_far:
                expl = explanator.attribute(back_test.to(device), background.to(device), target=c).cpu()
                explanations.append(expl)

            torch.save([explanations, back_test, back_test_targets], os.path.join(folder_paths[strategy_name], f'explanations_and_examples_{i}.pt'))
            torch.save(model.state_dict(), os.path.join(folder_paths[strategy_name], f'model{i}.pt'))
            plot_explanations_grid(explanations, back_test,
                           folder_paths[strategy_name], name=f'{strategy_name}_{args["dataset"]}_{i}',
                           num_plots=args['num_plots'])

        with open(os.path.join(folder_paths[strategy_name], 'metrics.pickle'), 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_mb_size', type=int, default=64)

    parser.add_argument('--joint_lr', type=float, default=1e-3)
    parser.add_argument('--joint_epochs', type=int, default=10)
    parser.add_argument('--joint_train_mb_size', type=int, default=64)

    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--gamma_min', type=float, default=1)
    parser.add_argument('--gamma_max', type=float, default=1)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_max', type=float, default=0.1)

    parser.add_argument('--spectral_radius', type=float, default=0.99)
    parser.add_argument('--input_scaling', type=float, default=1)

    parser.add_argument('--hidden_size', type=int, default=2000)
    parser.add_argument('--num_background', type=int, default=600)
    parser.add_argument('--replay_mem_size', type=int, default=100)
    parser.add_argument('--num_test_per_class', type=int, default=50)
    parser.add_argument('--num_plots', type=int, default=6)
    parser.add_argument('--speech_root', type=str, default='/home/a.cossu/synthethic_speech_commands_preprocessed')
    parser.add_argument('--experiment_folder', type=str, default='/home/a.cossu/explainable-continual-learning/results')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'speech'], default='mnist')
    parser.add_argument('--explanator', type=str, choices=['shap', 'lift'], default='shap')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn', 'rnn', 'ron', 'esn'], default='mlp')
    args = vars(parser.parse_args())
    main(args)
