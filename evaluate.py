from train import train_lossi, dev_lossi, train_accuracyi, dev_accuracyi
import matplotlib.pyplot as plt

def plot_training_progress():
  plt.figure(figsize=(12, 5))
  benchmarks = ['Loss', 'Accuracy']
  a = [[train_lossi, dev_lossi], [train_accuracyi, dev_accuracyi]]
  for i, benchmark in enumerate(benchmarks):
    plt.subplot(1, 2, i + 1)
    plt.plot(a[i][0], label='Train ' + benchmark, color='blue')
    plt.plot(a[i][1], label='Dev ' + benchmark, color='red')
    plt.title(f'{benchmark} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(benchmark)
    plt.legend()
  plt.tight_layout()
  plt.savefig('training_progress_separate.png')
  plt.show()
  