n_head = 8
n_embb = 512
block_size = 32
dropout = 0.3
device = 'cuda:2'
n_layer = 8
num_class = 4

lr = 1e-4
max_epoch = 100
data_size = 120000
batch_size = 32
num_batch = 120000//32
print(f'Using device = {device}')