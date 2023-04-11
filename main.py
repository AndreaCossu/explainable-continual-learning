import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from xai import compute_shap

def naive_scifar10():
    args = {
        "cuda": 0,
        "lr": 0.05,
        "epochs": 3,
        "train_mb_size": 64,
        "seed": 0,
    }
    fixed_class_order = np.arange(10)
    device = torch.device(
        f"cuda:{args['cuda']}" if torch.cuda.is_available() and args['cuda'] >= 0 else "cpu"
    )
    benchmark = SplitCIFAR10(
        5,
        return_task_id=False,
        seed=args['seed'],
        fixed_class_order=fixed_class_order,
        train_transform=transforms.ToTensor(),
        eval_transform=transforms.ToTensor(),
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )
    model = SlimResNet18(nclasses=10)
    optimizer = SGD(model.parameters(), lr=args['lr'])
    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]
    training_metrics = []
    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
    ]
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )
    plugins = [
    ]
    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        train_epochs=args['epochs'],
        evaluator=evaluator,
        device=device,
        train_mb_size=args['train_mb_size'],
        eval_mb_size=64,
    )
    results = []
    for experience in benchmark.train_stream:
        cl_strategy.train(
            experience,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )
        results.append(cl_strategy.eval(benchmark.test_stream))
    return results


if __name__ == "__main__":
    res = naive_scifar10()
    print(res)
