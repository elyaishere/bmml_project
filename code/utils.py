import os
from torch.nn.modules.batchnorm import _BatchNorm
from IPython.display import clear_output
import matplotlib.pyplot as plt


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def print_metrics(loss, acc):
    epoch = len(loss['train'])
    clear_output(True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Epoch #{epoch}')
    ax[0].set_title('loss')
    ax[0].plot(loss['train'], 'r.-', label='train')
    ax[0].plot(loss['test'], 'g.-', label='test')
    ax[0].legend()
    ax[1].set_title('accuracy')
    ax[1].plot(acc['train'], 'r.-', label='train')
    ax[1].plot(acc['test'], 'g.-', label='test')
    ax[1].legend()
    plt.show()


def create_checkoint_dir(root, model_name, dataset_name):
    checkpoint_dir = os.path.join(root, model_name, dataset_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
