import json
import os
import shutil

import torch

from cnn.AverageMeter import AccuracyHistory, LossHistory


def save_model(path_to_save, model, optimizer, epoch):
    _clear_folder(path_to_save)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    all_state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }

    torch.save(all_state, path_to_save)


def save_history(accuracy_history: AccuracyHistory,
                 loss_history: LossHistory):
    loss = json.dumps({'losses': loss_history.history})
    acc = json.dumps({'accuracy': accuracy_history.history})

    with open('model-accuracy.json', 'w') as outfile:
        json.dump(acc, outfile, indent=4, sort_keys=True, separators=(',', ': '))
        print(f'Save model loss to {os.path.abspath("model-accuracy.json")}')

    with open('model-loss.json', 'w') as outfile:
        json.dump(loss, outfile, indent=4, sort_keys=True, separators=(',', ': '))
        print(f'Save model accuracy to {os.path.abspath("model-loss.json")}')


def _clear_folder(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
