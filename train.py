import torch.optim as optim
from model import GPT
from config import *
from model import vocab_size
import torch.nn as nn
from utils import get_logger, get_lr
from dataset import train_data_loader, dev_data_loader
import torch
import time

model = GPT().to(device)
cri = nn.CrossEntropyLoss()
optimizer = model.config_optimizer(weight_decay = 0.1, lr = 6e-4)

train_lossi = []
dev_lossi = []
train_accuracyi = []
dev_accuracyi = []
logger = get_logger()

start_epoch = 0



min_loss = 10
for epoch in range(start_epoch, max_epoch):
  start = time.time()
  model.train()
  print(f'epoch: {epoch + 1}')
  dev_loss_epoch = 0
  dev_accuracy_epoch = 0
  train_loss_epoch = 0
  train_accuracy_epoch = 0
  total_train = 0
  total_dev = 0
  lr = get_lr(epoch)
  print(f'lr={lr}')
  for para_group in optimizer.param_groups:
    para_group['lr'] = lr
  
  for x, y in train_data_loader:
    
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    optimizer.zero_grad()
    loss = cri(output, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    with torch.no_grad():
      train_accuracy_epoch += (torch.argmax(output, dim = 1) == y).sum().item()
      train_loss_epoch += loss.item() * x.shape[0]
      total_train += x.shape[0]
  
  train_accuracy_epoch /= total_train
  train_loss_epoch /= total_train
  
  model.eval()
  end = time.time()
  epoch_time = end - start
  with torch.no_grad():
    for x, y in dev_data_loader:
      x = x.to(device)
      y = y.to(device)
      output = model(x)
      loss = cri(output, y)
      dev_accuracy_epoch += (torch.argmax(output, dim = 1) == y).sum().item()
      dev_loss_epoch += loss.item() * x.shape[0]
      total_dev += x.shape[0]

    dev_accuracy_epoch /= total_dev
    dev_loss_epoch /= total_dev
  train_lossi.append(train_loss_epoch)
  dev_lossi.append(dev_loss_epoch)
  train_accuracyi.append(train_accuracy_epoch)
  dev_accuracyi.append(dev_accuracy_epoch)

  
  logger.info(f'Epoch {epoch + 1}: Train Loss = {train_loss_epoch:.4f}, Dev Loss = {dev_loss_epoch:.4f}, Train Accuracy = {train_accuracy_epoch:.4f}, Dev Accuracy = {dev_accuracy_epoch:.4f}, Epoch {epoch + 1} get {(epoch_time / 60):.2f}p to train')
torch.save({
      'epoch': epoch + 1,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': dev_loss_epoch,
  }, 'checkpoint.pth')