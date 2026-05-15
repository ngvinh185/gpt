from collections import Counter
from config import block_size, batch_size
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
import pandas as pd



def dele(data, p = 0.1):
  words = data.split()
  a = torch.randn(len(words))
  a_norm = (a - a.min()) / (a.max() - a.min())
  new_words = []
  new_words += [word for idx, word in enumerate(words) if a_norm[idx] >= p]
  return " ".join(new_words)

def swap(data):
  a = torch.randint(1, 10, (1,))
  words = data.split()
  n = len(words)
  for _ in range(a):
    i1, i2 = torch.randint(0, n, (2,))
    words[i1], words[i2] = words[i2], words[i1]
  return " ".join(words)

def augment_data(data):
  choice = torch.randint(1, 3, (1,))
  if choice == 1:
    new_data = dele(data)
  else: new_data = swap(data)
  return new_data

_dict = {}
counter = Counter()
def no_name(df):
  global _dict
  for t in df['text']:
    counter.update(t.split())
  a = [word for word, fre in counter.items() if fre > 5]
  _dict = {w:i for i, w in enumerate(a)}
  _dict['UNK'] = len(a)
  _dict['PAD'] = len(a) + 1



def handle_data(x):
  y = []
  for w in x:
    if w not in _dict: w = 'UNK'
    y.append(_dict[w])
  if len(y) > block_size: y = y[:block_size]
  else: y += [_dict['PAD'] for _ in range(block_size-len(y))]

  return y

class dataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    tmp = torch.randint(1, 3, (1, ))
    if tmp == 1: xi = handle_data(self.x[idx])
    else:
      xi = augment_data(self.x[idx])
      xi = handle_data(xi)
    yi = self.y[idx] 
    return torch.tensor(xi, dtype = torch.long), torch.tensor(yi, dtype=torch.long)

ds = load_dataset("fancyzhx/ag_news")
df_train = pd.DataFrame(ds['train'])
df_dev = pd.DataFrame(ds['test'])
no_name(df_train)
train_data = dataset(df_train['text'].tolist(), df_train['label'].tolist())
dev_data = dataset(df_dev['text'].tolist(), df_dev['label'].tolist())
train_data_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=2)
dev_data_loader = DataLoader(dev_data, batch_size, shuffle = False, num_workers=2)