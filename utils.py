import os
import logging
import math
def get_logger(logs_dir = 'logs'):
  os.makedirs(logs_dir, exist_ok = True)
  logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
      logging.FileHandler(os.path.join(logs_dir, 'training.log')),
      logging.StreamHandler()
    ]
  )
  return logging.getLogger()

base_epoch = 0
max_epoch = 70
start_lr = 6e-4
end_lr = 1e-6

def get_lr(epoch):
  if epoch < max_epoch:
    decay_ratio = (epoch - base_epoch) / (max_epoch - base_epoch)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return end_lr + coeff * (start_lr - end_lr)
  else:
    return end_lr
    