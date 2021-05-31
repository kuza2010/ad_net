import json

from matplotlib import pyplot as plt

losses = {
    'iteration': [],
    'values': []
}

acc = {
    'epoch': [],
    'accuracy': []
}

with open('../acc_and_loss/model-loss-tmp.json', 'r') as outfile:
    data = outfile.read().replace("\\", "")
    data = data[1:-1]
    data = json.loads(data)
    for entry in data['losses']:
        losses['iteration'].append(entry['iteration'])
        losses['values'].append(round(entry['val'], 3))

    print(losses)

with open('../acc_and_loss/model-accuracy-tmp.json', 'r') as outfile:
    data = outfile.read().replace("\\", "")
    data = data[1:-1]
    data = json.loads(data)
    for entry in data['accuracy']:
        acc['epoch'].append(entry['epoch'])
        acc['accuracy'].append(round(entry['accuracy'], 3))

    print(acc)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Iteration and accuracy')

ax1.plot(acc['epoch'], acc['accuracy'])
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy")

ax2.plot(losses['iteration'], losses['values'])
ax2.set_xlabel("iteration")
ax2.set_ylabel("loss")

ax1.grid()
ax2.grid()
plt.show()
