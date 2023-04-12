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
from avalanche.training.supervised import Naive, Replay
from utils import plot_shap, IncrementalMLP, get_shap_background_and_test, \
    ReducedResNet18, SequenceClassifier, create_speech_benchmark
import json
from captum.attr import GradientShap


def main(args):
    exp_name = f"{args['strategy']}_{args['dataset']}" if not args["test"] else "test"
    folder_path = os.path.join(args['experiment_folder'], exp_name)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'args.json'), 'w') as f:
        json.dump(args, f)

    device = torch.device(
        f"cuda:{args['cuda']}" if torch.cuda.is_available() and args['cuda'] >= 0 else "cpu"
    )

    if args['dataset'] == 'cifar':
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
        model = ReducedResNet18()
        optimizer = SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    elif args['dataset'] == 'mnist':
        benchmark = SplitMNIST(5, return_task_id=False,
                               fixed_class_order=list(range(10)),
                               shuffle=True,
                               class_ids_from_zero_in_each_exp=False)
        model = IncrementalMLP(num_classes=10, input_size=28*28, hidden_size=256, hidden_layers=1)
        optimizer = SGD(model.parameters(), lr=args['lr'])
    elif args['dataset'] == 'speech':
        benchmark = create_speech_benchmark(num_words=10, test_split=0.2)
        model = SequenceClassifier(input_size=40, hidden_size=256, rnn_layers=2, batch_first=True)
        optimizer = Adam(model.parameters(), lr=args['lr'])
    else:
        raise ValueError("Unrecognized dataset name")

    shap = GradientShap(model)
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

    if args['strategy'] == 'naive':
        cl_strategy = Naive(
            model=model,
            optimizer=optimizer,
            train_epochs=args['epochs'],
            evaluator=evaluator,
            device=device,
            train_mb_size=args['train_mb_size'],
            eval_mb_size=64,
        )
    elif args['strategy'] == 'replay':
        cl_strategy = Replay(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_epochs=args['epochs'],
            evaluator=evaluator,
            device=device,
            train_mb_size=args['train_mb_size'],
            eval_mb_size=64,
            mem_size=500,
        )
    else:
        raise ValueError("Unrecognized strategy name.")

    results = []

    shap_background, shap_test, shap_test_targets = get_shap_background_and_test(
        dataset=benchmark.test_stream[0].dataset,
        num_background=args['num_background'],
        num_inputs_per_class=args['num_test_per_class'],
        classes=[0, 1])

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

        # explanations is a list of (n_classes) elements. Each element is a numpy array of shape (num_test, input_shape),
        # where input_shape is (1, 28, 28) for MNIST, (3, 32, 32) for CIFAR10, (101, 40) for Speech.
        explanations = []
        for c in classes_so_far:
            expl = shap.attribute(shap_test.to(device), shap_background.to(device), target=c)
            explanations.append(expl.cpu())

        torch.save([explanations, shap_test], os.path.join(folder_path, f'explanations_and_examples_{i}.pt'))
        torch.save(model.state_dict(), os.path.join(folder_path, f'model{i}.pt'))
        plot_shap(explanations, shap_test, folder_path, name=f'{args["strategy"]}_{args["dataset"]}_{i}')

    with open(os.path.join(folder_path, 'metrics.pickle'), 'wb') as f:
        pickle.dump(results, f)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_mb_size', type=int, default=64)
    parser.add_argument('--num_background', type=int, default=700)
    parser.add_argument('--num_test_per_class', type=int, default=3)
    parser.add_argument('--experiment_folder', type=str, default='/disk3/a.cossu/explainable-continual-learning/results')
    parser.add_argument('--strategy', type=str, default='naive')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--test', action="store_true")
    args = vars(parser.parse_args())
    res = main(args)
    print(res)