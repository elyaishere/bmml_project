import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .utils import *


def train(optimizer, scheduler, model, dataset, num_epochs,
          label_smoothing=0.0, training_dir='runs', device='cpu'):
  checkpoint_dir = os.path.join(training_dir, model._model_name, dataset._dataset_name)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  best_test_loss, best_test_acc = None, None
  loss_hist = {'train': [], 'test': []}
  acc_hist = {'train': [], 'test': []}

  for epoch in range(num_epochs):
    running_loss = 0.0
    running_accuracy = 0.0
    model.train()

    for batch in tqdm(dataset.train):
        if dataset._batch_augmentation is not None:
            inputs, targets = (b.to(device) for b in dataset._batch_augmentation(batch))
        else:
            inputs, targets = (b.to(device) for b in batch)

        enable_running_stats(model)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
        loss.backward()
        running_loss += loss.item()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model)
        F.cross_entropy(model(inputs), targets, label_smoothing=label_smoothing).backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            running_accuracy += (torch.argmax(outputs.data, 1) == targets).sum().item()
    scheduler.step()

    loss_hist['train'] += [running_loss / len(dataset.train)]
    acc_hist['train'] += [100 * running_accuracy / dataset.train_size]

    running_loss = 0.0
    running_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset.test):
            inputs, targets = (b.to(device) for b in batch)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs.data, 1) == targets).sum().item()
          
        loss_hist['test'] += [running_loss / len(dataset.test)]
        acc_hist['test'] += [100 * running_accuracy / dataset.test_size]

        if best_test_loss is None or running_loss < best_test_loss:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, os.path.join(checkpoint_dir, 'best_test_loss.pth'))
          best_test_loss = running_loss

        if best_test_acc is None or running_accuracy > best_test_acc:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, os.path.join(checkpoint_dir, 'best_test_acc.pth'))
          best_test_acc = running_accuracy

    print_metrics(loss_hist, acc_hist)
