import matplotlib.pyplot as plt
import os
import numpy as np
import shap
from torch.utils.data import DataLoader


def compute_shap(model, test_dataset, device, num_background=700, num_inputs=5):
    batch_size = num_background + num_inputs
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for batch in test_dataloader:
        x, _, _ = batch
        break

    background_inputs = x[:num_background]
    test_inputs = x[num_background:num_background+num_inputs]
    e = shap.DeepExplainer(model, background_inputs.to(device))
    shap_values = e.shap_values(test_inputs.to(device))
    return shap_values, test_inputs


def plot_shap(shap_values, test_inputs, save_path, name='test'):
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_inputs.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)
    plt.savefig(os.path.join(save_path, f'{name}.png'))
