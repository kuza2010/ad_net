import json
import os
import shutil

import torch

from AverageMeter import AccuracyHistory, LossHistory


def save_model(path_to_save, model, optimizer, epoch):
    clear_folder(path_to_save)

    all_state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }

    torch.save(all_state, path_to_save)


def save_history(accuracy_history: AccuracyHistory,
                 loss_history: LossHistory):
    loss = json.dumps(loss_history.history, indent=2)
    acc = json.dumps(accuracy_history.history, indent=2)

    with open('model-accuracy.json', 'w') as outfile:
        json.dump(acc, outfile)
        print(f'Save model loss to model-accuracy.json')

    with open('model-loss.json', 'w') as outfile:
        json.dump(loss, outfile)
        print(f'Save model accuracy to model-loss.json')


def clear_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
