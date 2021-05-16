# Average meter
# https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


class LossHistory:
    def __init__(self):
        self.history = []

    def __str__(self) -> str:
        return '; '.join(
            [f'iteration [#{entry["iteration"]}], loss {entry["val"]:.4f}' for entry in self.history]
        )

    def reset(self):
        self.history.clear()

    def append(self, val, iteration):
        self.history.append({'val': val, 'iteration': iteration})


class AccuracyHistory:
    def __init__(self):
        self.history = []

    def __str__(self) -> str:
        return '; '.join(
            [f'epoch [#{entry["epoch"]}], accuracy {entry["accuracy"]:.4f}' for entry in self.history]
        )

    def reset(self):
        self.history.clear()

    def append(self, epoch, accuracy):
        self.history.append({'epoch': epoch, 'accuracy': accuracy})
