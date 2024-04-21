import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

statistics = np.load('training_statistics.npy', allow_pickle=True).item()

train_losses = statistics['train_losses']
val_losses = statistics['val_losses']
val_accuracies = statistics['val_accuracies']

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'firebrick'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(train_losses, label='Training Loss', linestyle='dotted', color='mediumblue')
ax1.plot(val_losses, label='Validation Loss', linestyle='dotted', color=color)
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a twin axis sharing the same x-axis.
ax2 = ax1.twinx()

ax2.set_ylabel('Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy', color='firebrick')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

ax1.set_title('ResNet18 Training and Validation Loss with Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')

# Set the x-axis to display only integer values.
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Modify x-axis tick labels to display values incremented by 1.
current_labels = [label.get_text() for label in ax1.get_xticklabels()]

# Replace '−' with '-' in each label and convert to integers.
new_labels = [int(label.replace('−', '-')) for label in current_labels]

new_labels = [str(label + 1) for label in new_labels]
ax1.set_xticklabels(new_labels)

fig.tight_layout()

plt.savefig('resnet_stats.png')
plt.show()
