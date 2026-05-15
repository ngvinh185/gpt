from config import n_head, n_embb, block_size, dropout, device, n_layer, num_class
from dataset import _dict
import torch
import torch.nn as nn
import torch.nn.functional as F
vocab_size = len(_dict)
# class Head(nn.Module):
#   def __init__(self, head_size):
#     super().__init__()
#     self.head_size = head_size
#     self.query = nn.Linear(n_embb, head_size)
#     self.key = nn.Linear(n_embb, head_size)
#     self.value = nn.Linear(n_embb, head_size)
#     self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
#   def forward(self, x):
#     B, T, C = x.shape
#     q = self.query(x)
#     k = self.key(x)
#     wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
#     wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
#     wei = F.softmax(wei, dim=-1) # (B, T, T)
#     out = wei @ self.value(x)
#     return out

# class MultiHeadAttention(nn.Module):
#   def __init__(self, n_head, n_embb):
#     super().__init__()
#     head_size = n_embb // n_head
#     self.mha = nn.ModuleList([Head(head_size) for _ in range(n_head)])
#     self.ln = nn.LayerNorm(n_embb)
#     self.proj = nn.Linear(n_embb, n_embb)
#     self.dropout = nn.Dropout(dropout)
#   def forward(self, x):
#     x = torch.cat([h(x) for h in self.mha], -1)
#     x = self.ln(x)
#     x = self.proj(x)
#     x = self.dropout(x)
#     return x

# class FeedForward(nn.Module):
#   def __init__(self, n_embb):
#     super().__init__()
#     self.fc1 = nn.Linear(n_embb, 4 * n_embb)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(4 * n_embb, n_embb)
#     self.dropout = nn.Dropout(dropout)
#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     x = self.dropout(x)
#     return x

# class Block(nn.Module):
#   def __init__(self, n_embb, n_head):
#     super().__init__()
#     self.multihead = MultiHeadAttention(n_head, n_embb)
#     self.ffwd = FeedForward(n_embb)
#     self.ln1 = nn.LayerNorm(n_embb)
#     self.ln2 = nn.LayerNorm(n_embb)

#   def forward(self, x):
#     x = x + self.multihead(self.ln1(x))
#     x = x + self.ffwd(self.ln2(x))
#     return x
  
# class BiagramModel(nn.Module):
#   def __init__(self, n_embb, n_head):
#     super().__init__()
#     self.pos_table_emb = nn.Embedding(block_size, n_embb)
#     self.token_table_emb = nn.Embedding(vocab_size, n_embb)
#     self.blocks = nn.Sequential(*[Block(n_embb, n_head) for _ in range(n_layer)])
    
#     # self.ln2 = nn.Linear(n_embb, vocab_size)
#     self.conv1 = nn.Sequential(nn.Conv1d(in_channels = n_embb, out_channels = n_embb, kernel_size=3, stride=2), nn.ReLU())
#     self.conv2 = nn.Sequential(nn.Conv1d(in_channels = n_embb, out_channels = n_embb, kernel_size=7, stride=2), nn.ReLU())
#     self.ln = nn.LayerNorm(n_embb)
#     self.fc1 = nn.Sequential(nn.Linear(n_embb * 5, n_embb), nn.ReLU(), nn.Dropout(dropout))
#     self.fc2 = nn.Linear(n_embb, num_class)
#     self.apply(self._init_weights)

  
#   def forward(self, x):
#     B, T = x.shape
#     p = self.pos_table_emb(torch.arange(T, device=device))
#     t = self.token_table_emb(x)
#     x = p + t
#     x = self.blocks(x)
#     # 
    
    
#     B, T, C = x.shape
#     x = x.transpose(1, 2)
#     x = self.conv2(self.conv1(x))
#     x = x.transpose(1, 2)
#     x = self.ln(x)
#     x = x.view(B, C * 5)
#     logits = self.fc2(self.ln(self.fc1(x)))
    
#     return logits
#   def _init_weights(self, module):
#     if isinstance(module, nn.Linear):
#       nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
#       if module.bias is not None:
#         nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#       nn.init.normal_(module.weight, mean = 0.0, std = 0.02)


class CausualAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_att = nn.Linear(n_embb, n_embb*3)
    self.c_proj = nn.Linear(n_embb, n_embb)
    self.c_proj.NANOGPT_SCALE_INIT = 1
  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.c_att(x).split(n_embb, dim = 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_fc = nn.Linear(n_embb, n_embb*4)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(n_embb * 4, n_embb)
    self.c_proj.NANOGPT_SCALE_INIT = 1
  def forward(self, x):
    x = self.c_proj(self.gelu(self.c_fc(x)))
    return x
class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.layernorm1 = nn.LayerNorm(n_embb)
    self.layernorm2 = nn.LayerNorm(n_embb)
    
    self.att = CausualAttention()
    self.mlp = MLP()
  
  def forward(self, x):
    x = x + self.att(self.layernorm1(x))
    x = x + self.mlp(self.layernorm2(x))
    return x
    
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(vocab_size, n_embb),
      wpe = nn.Embedding(block_size, n_embb),
      h = nn.ModuleList([Block() for _ in range(n_layer)]),
      ln_f = nn.LayerNorm(n_embb)
    ))

    # self.lm_head = nn.Linear(n_embb, vocab_size)
    # self.transformer.wte.weight = self.lm_head.weight
    
    self.fc1 = nn.Sequential(nn.Linear(n_embb), nn.BatchNorm1d(n_embb), nn.ReLU(), nn.Dropout(dropout))
    self.fc2 = nn.Linear(n_embb, num_class)
    
    self.apply(self._init_weight)
  def _init_weight(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module,  'NANOGPT_SCALE_INIT'):
        std *= (2 * n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean = 0, std = std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  def config_optimizer(self, weight_decay, lr):
    _parameters = {n:par for n, par in self.named_parameters()}
    _parameters = {n:par for n, par in _parameters.items() if par.requires_grad}
    parameters_decay = [par for n, par in _parameters.items() if par.dim() >= 2]
    parameters_no_decay = [par for n, par in _parameters.items() if par.dim() < 2]
    optim_groups = [
      {'params': parameters_decay, 'weight_decay': weight_decay},
      {'params': parameters_no_decay, 'weight_decay': 0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return optimizer
  def forward(self, x):
    B, T = x.shape
    x_emb = self.transformer.wte(x)
    p_emb = self.transformer.wpe(torch.arange(T, device=device))
    x = x_emb + p_emb
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    x = self.fc1(x)
    x = self.fc2(x)
    return x
    
    