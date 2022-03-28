
"""# Inference

## Dataset of inference
"""

data_dir = "./Dataset"
output_name = "ensemble_80_80_3_25_finish"

import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn as nn
import torchaudio.models as models

class Classifier(nn.Module):
	def __init__(self, d_model=80, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)

		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		# self.encoder_layer = nn.TransformerEncoderLayer(
		# 	d_model=d_model, dim_feedforward=256, nhead=2
		# )
		# self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=5)

		self.conformer = models.Conformer(
			input_dim=d_model,
			num_heads=1,
			ffn_dim=80,
			num_layers=3,
			depthwise_conv_kernel_size=25,
			# dropout=dropout,
		)

		self.att_pool = nn.Sequential(
			nn.Linear(d_model, 1),
			nn.Softmax(dim=1)
		)

		# Project the the dimension of features from d_model into speaker nums.
		# self.ffn = nn.Sequential(
		# 	nn.Linear(d_model, d_model),
		# 	nn.ReLU(),
		# 	# nn.Dropout(p=dropout),
		# )

		self.s = 30.0
		self.m = 0.4
		self.pred = nn.Linear(d_model, n_spks, bias=False)

	def forward(self, mels, labels, lengths):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""

		# out: (batch size, length, d_model)
		out = self.prenet(mels)

		####
		# Use Transformer Encoder
		####
		# out: (length, batch size, d_model)
		# out = out.permute(1, 0, 2)

		# The encoder layer expect features in the shape of (length, batch size, d_model).
		# out = self.encoder_layer(out)
		# out = self.encoder(out)

		# out: (batch size, length, d_model)
		# out = out.transpose(0, 1)

		####
		# Use Conformer
		####
		out, _ = self.conformer(out, lengths)
		# print(f'out.shape = {out.shape}')

		####
		# mean pooling
		####
		# stats = out.mean(dim=1)

		# out: (batch, n_spks)
		# out = self.pred_layer(stats)

		####
		# self attention pooling
		####
		pool = self.att_pool(out)
		pool = torch.transpose(pool, 1, 2)
		# print(f'pool.shape = {pool.shape}')
		rep = torch.matmul(pool, out)
		# print(f'rep.shape = {rep.shape}')
		rep = torch.squeeze(rep, dim=1)
		
		# x = self.ffn(rep)
		x = rep

		out = self.pred(x)

		#
		# AMSoftmax Loss
		#
		if labels is not None:
			for W in self.pred.parameters():
				W = F.normalize(W, dim=1)
			
			x = F.normalize(x, dim=1)

			wf = self.pred(x)
			numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
			excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
			denominator = torch.log(torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1))
			L = numerator - denominator
			loss = -torch.mean(L)

			return out, loss
		
		return out

class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)

"""## Main funcrion of Inference"""

import json
import csv
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info]: Use {device} now!")

mapping_path = Path(data_dir) / "mapping.json"
mapping = json.load(mapping_path.open())

dataset = InferenceDataset(data_dir)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=inference_collate_batch,
)
print(f"[Info]: Finish loading data!",flush = True)

speaker_num = len(mapping["id2speaker"])

model_1 = Classifier(n_spks=speaker_num).to(device)
model_1.load_state_dict(torch.load('./AMSoftmax_CV_seed_326.ckpt'))
model_1.eval()

model_2 = Classifier(n_spks=speaker_num).to(device)
model_2.load_state_dict(torch.load('./AMSoftmax_CV_seed_1121.ckpt'))
model_2.eval()

model_3 = Classifier(n_spks=speaker_num).to(device)
model_3.load_state_dict(torch.load('./AMSoftmax_CV_seed_1121326.ckpt'))
model_3.eval()

best_models = [model_1, model_2, model_3]

print(f"[Info]: Finish creating model!",flush = True)	

pbar = tqdm(dataloader, desc="Test", unit=" uttr")

results = [["Id", "Category"]]
with torch.no_grad():
    for feat_paths, mels in pbar:
        outs = None
        for best_model in best_models:
            mels = mels.to(device)
            # print(mels.shape)
            lengths = torch.LongTensor([mels.shape[1]]).to(device)
            if outs is None:
                outs = best_model(mels, None, lengths)
            else:
                outs += best_model(mels, None, lengths)

        preds = outs.argmax(1).cpu().numpy()
        for feat_path, pred in zip(feat_paths, preds):
            results.append([feat_path, mapping["id2speaker"][str(pred)]])

with open(f"./{output_name}.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)

