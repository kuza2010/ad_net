import random
import time

# numpy
import numpy as np
# torch
import torch
from torch import nn
from torchvision.transforms import transforms

# network and periphery
from cnn import ADNetFactory, network_utils, file_utils
# constants
from cnn.ADTrainTransformation import ADTrainTransformation, ToTensor
from cnn.AverageMeter import AverageMeter, LossHistory, AccuracyHistory
from cnn.DeviceProvider import DeviceProvider
from cnn.ad_net_constants import DATA_LOADER_BATCH_SIZE, WEIGHT_DECAY, \
    LEARNING_RATE, MOMENTUM, DECAY_EPOCH, LR_GAMMA, EPOCH

MODEL_STATE_RELATIVE_PATH = './model/state/'


def _main_decorator(func):
    def seed_wrapper():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def device_wrapper():
        print(f'Using ***{device.device_info}*** device.')

    def wrapper():
        seed_wrapper()

        print('<===============     Start   ===============>\n')
        device_wrapper()
        func()
        print('\n<===============      End     ===============>')

    return wrapper


def _provide_data_loader_kwargs():
    return {'num_workers': 1, 'pin_memory': True}


def train_model(model, train_loader, optimizer, epoch, tlh):
    _time_meter = AverageMeter()
    _train_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    model.train()

    for batch_num, entry in enumerate(train_loader):
        batch_time = time.time()
        images, target = entry['images'], entry['label']

        shape = list(images.size())  # images.size() = [16, 2, 1, 256, 256]
        images = images.reshape(shape[0] * shape[1], *shape[2:])  # images.size() = [32, 1, 256, 256]
        target = target.reshape(-1)

        images = images.to(device.device)
        target = target.to(device.device)

        # clear gradients
        optimizer.zero_grad()
        # forward propagation
        output = model(images)
        # find the loss
        loss = criterion(output, target)
        # compute gradient
        loss.backward()
        # update weights
        optimizer.step()
        # compute loss

        _train_loss.update(loss.item(), images.size(0))
        _time_meter.update(time.time() - batch_time)
        print(f'Epoch: [{epoch + 1}][{batch_num + 1}/{len(train_loader)}]\t'
              f'Mini-batch time {_time_meter.val:.3f} avg({_time_meter.avg:.3f})\t'
              f'Loss {_train_loss.val:.4f} ({_train_loss.avg:.4f})\t')

    tlh.append(_train_loss.avg, epoch)


def evaluate_model(model, m_device, eval_loader, is_validation=False,
                   epoch=None, optimizer=None, best_acc=None, vah=None):
    correct = 0.0
    model.eval()

    # do evaluation
    with torch.no_grad():
        for entry in eval_loader:
            images, target = entry['images'], entry['label']

            shape = list(images.size())
            images = images.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)

            images = images.to(m_device.device)
            target = target.to(m_device.device)

            # forward propagation
            output = model(images)
            # get predictions from the maximum value
            prediction = output.max(1, keepdim=True)
            prediction_tensor = prediction[1]
            correct += prediction_tensor.eq(target.view_as(prediction_tensor)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)
    if is_validation:
        # append to history
        vah.append(epoch, accuracy)
        if accuracy > best_acc:
            file_utils.save_model(MODEL_STATE_RELATIVE_PATH, model, optimizer, epoch)

    print('-' * 25)
    print(f'Epoch {epoch}\t'
          f'Eval accuracy: {accuracy:.4f}\t'
          f'Best accuracy: {best_acc}')
    print('-' * 25)

    return best_acc


@_main_decorator
def main():
    # create dataset
    test_data_set, validation_data_set, train_data_set = network_utils.get_data_set(
        m_seed=seed,
        train_transforms=transforms.Compose([ADTrainTransformation(), ToTensor()]),
        test_transforms=transforms.Compose([ToTensor()]),
        validation_transforms=transforms.Compose([ToTensor()])
    )
    # create data loaders
    test_dataloader, validation_dataloader, train_dataloader = network_utils.get_data_loaders(
        train_data_set,
        validation_data_set,
        test_data_set,
        DATA_LOADER_BATCH_SIZE,
        _provide_data_loader_kwargs())

    # create model
    model = ADNetFactory.build(device)
    # create optimizer
    optimizer = network_utils.get_optimizer(model, WEIGHT_DECAY, LEARNING_RATE, MOMENTUM)
    # load prev network state
    start_epoch = network_utils.load_state(model, optimizer, MODEL_STATE_RELATIVE_PATH)
    # add learning scheduler
    scheduler = network_utils.get_learning_scheduler(optimizer, DECAY_EPOCH, LR_GAMMA)

    best_accuracy = 0.0
    train_loss_history = LossHistory()
    validation_accuracy_history = AccuracyHistory()

    for epoch in range(start_epoch, EPOCH):
        # train
        train_model(model, train_dataloader, optimizer, epoch, train_loss_history)
        # validate
        best_accuracy = evaluate_model(model=model,
                                       m_device=device,
                                       eval_loader=validation_dataloader,
                                       is_validation=True,
                                       epoch=epoch,
                                       optimizer=optimizer,
                                       best_acc=best_accuracy,
                                       vah=validation_accuracy_history)
        scheduler.step()

    # load best network parameter to test
    network_utils.load_state(model, optimizer, MODEL_STATE_RELATIVE_PATH)
    evaluate_model(model, device, test_dataloader, False)
    # save history to file
    file_utils.save_history(validation_accuracy_history, train_loss_history)
    # print(model.info((1, 256, 256)))


if __name__ == '__main__':
    device = DeviceProvider()
    seed = random.randint(0, 20051997)

    main()
