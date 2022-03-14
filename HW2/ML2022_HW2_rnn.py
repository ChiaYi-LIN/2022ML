#%% [markdown]
# **Helper functions to pre-process the training data from raw MFCC features of each utterance.**
# 
# A phoneme may span several frames and is dependent to past and future frames. \
# Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.
# 
# Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)

#%%
import os
import random
import pandas as pd
import torch
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    # print(feat.shape)
    # print(feat)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        # print('x[-1] =', x[-1])
        right = x[-1].repeat(n, 1)
        # print('right', right)
        left = x[n:]
        # print('left =', left)
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    # print('seq_len =', seq_len)
    # print('feature_dim =', feature_dim)
    x = x.repeat(1, concat_n) 
    # print('x =', x.shape)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    # print('x =', x.shape)
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)
    # print('new x =', x.shape)
    return x.permute(1, 0, 2).view(seq_len, concat_n, feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    # max_len = 3000000
    X = []
    if mode != 'test':
      y = []

    # idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt')).tolist()
        # print('feat =', feat.shape)
        
        # cur_len = len(feat)
        # feat = concat_feat(feat, concat_nframes)
        # if mode != 'test':
        #   label = torch.LongTensor(label_dict[fname])
        if mode != 'test':
          label = label_dict[fname]

        # X[idx: idx + cur_len, :] = feat
        X.append(feat)
        if mode != 'test':
        #   y[idx: idx + cur_len] = label
            y.append(label)

        # idx += cur_len

    # X = X[:idx, :]
    # if mode != 'test':
    #   y = y[:idx]

    if mode != 'test':
      df = {'X': X, 'y': y}
    else:
      df = {'X': X}
    df = pd.DataFrame(df)

    print(f'[INFO] {split} set')
    # print(df.head(5))
    # print(df.shape)
    if mode != 'test':
    #   print(y.shape)
      return df.X.values, df.y.values
    else:
      return df.X.values


#%% [markdown]
# ## Define Dataset

#%%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = y
        else:
            self.label = None
        
        self.cnt = 0
        for row in X:
            self.cnt += len(row)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return torch.FloatTensor(self.data[idx])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        seq_vec, seq_label, lengths = zip(*[
            (torch.FloatTensor(vec), torch.LongTensor(label), len(vec))
            for (vec, label) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
        ])

        padded_seq_vec = pad_sequence(seq_vec, batch_first=True, padding_value=0)
        # print(padded_seq_vec.shape)
        # print(torch.LongTensor(seq_label).shape)

        padded_sequ_label = pad_sequence(seq_label, batch_first=True, padding_value=-100)
        # print('padded_sequ_label', padded_sequ_label.shape)

        return padded_seq_vec, padded_sequ_label, torch.LongTensor(lengths)


#%% [markdown]
# ## Define Model

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout=0, bidirectional=False):
        super(Classifier, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=hidden_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
        
        if self.bidirectional:
            # self.fc = nn.Linear(2 * hidden_dim, output_dim) # bidirectional
            self.fc = nn.Sequential(
                # *[BasicBlock(2 * hidden_dim, 2 * hidden_dim, dropout) for _ in range(hidden_layers)],
                nn.Linear(2 * hidden_dim, output_dim)
            )
        else:
            # self.fc = nn.Linear(hidden_dim, output_dim)
            self.fc = nn.Sequential(
                # *[BasicBlock(hidden_dim, hidden_dim, dropout) for _ in range(hidden_layers)],
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x, lengths=None):
        # print('x =', x.shape)
        
        # pack sequence
        packed_x = rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # packed_out, h = self.gru(pack_embeds)
        packed_out, (h, c) = self.lstm(packed_x)

        out, _ = rnn.pad_packed_sequence(packed_out, batch_first=True)

        tag_space = self.fc(out) # tag_space = torch.Size([batch size, seq length, num_class])
        # print('tag_space = ', tag_space.shape)
        
        outputs = tag_space.transpose(-1, 1) # outputs = torch.Size([batch size, num_class, seq length])

        return outputs

#%% [markdown]
# ## Hyper-parameters

#%%
# data prarameters
concat_nframes = 3              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1121326                        # random seed
batch_size = 16                # batch size
num_epoch = 100                   # the number of training epoch
learning_rate = 1e-3          # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 # the input dim of the model, you should not change the value
hidden_layers = 8               # the number of hidden layers
hidden_dim = 128                # the hidden dim
dropout = 0.35
bidirectional = True

# scheduler
# step_size = 20

#%% [markdown]
# ## Prepare dataset and model

#%%
import gc

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)
print('train_set cnt =', train_set.cnt)
print('val_set cnt =', val_set.cnt)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=val_set.collate_fn)

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f'DEVICE: {device}')

#%%
import numpy as np

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%%
# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, dropout=dropout, bidirectional=bidirectional).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

#%% [markdown]
# ## Training

#%%
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train() # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels, lengths = batch
        features = features.to(device)
        labels = labels.to(device)
        # print('features =', features.shape)
        # print('labels =', labels.shape)
        # print('lengths =', lengths.shape)
        # print(lengths)
        
        optimizer.zero_grad() 
        outputs = model(features, lengths)
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        # print('train_pred =', train_pred.shape)
        for j in range(len(labels)):
            gt_vec = labels[j][:lengths[j]]
            pred_vec = train_pred[j][:lengths[j]]
            # print('gt_vec =', gt_vec.shape)
            # print(gt_vec)
            # print('pred_vec =', pred_vec.shape)
            # print(pred_vec)
            train_acc += (pred_vec == gt_vec).sum().item()
            
        train_loss += loss.item()
    
    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels, lengths = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features, lengths)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) 
                for j in range(len(labels)):
                    gt_vec = labels[j][:lengths[j]]
                    pred_vec = val_pred[j][:lengths[j]]
                    # print('gt_vec =', gt_vec.shape)
                    # print(gt_vec)
                    # print('pred_vec =', pred_vec.shape)
                    # print(pred_vec)
                    val_acc += (pred_vec == gt_vec).sum().item()
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_set.cnt, train_loss/len(train_loader), val_acc/val_set.cnt, val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/val_set.cnt))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/train_set.cnt, train_loss/len(train_loader)
        ))
    
    # scheduler.step()

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


#%%
del train_loader, val_loader
gc.collect()

#%% [markdown]
# ## Testing
# Create a testing dataset, and load model from the saved checkpoint.

#%%
# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

#%%
# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, dropout=dropout, bidirectional=bidirectional).to(device)
model.load_state_dict(torch.load(model_path))

#%% [markdown]
# Make prediction.

#%%
test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features, [features.shape[1]])
        # print('outputs = ', outputs.shape)
        
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        # print('test_pred = ', test_pred.shape)

        pred = np.append(pred, test_pred.cpu().numpy())
        # print('pred = ', pred.shape)


#%% [markdown]
# Write prediction to a CSV file.
# 
# After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.

#%%
with open('rnn_pred.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))


