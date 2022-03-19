_exp_name = 'resnet_18_34_50_ensemble'

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# "cuda" only when GPUs are available.
if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is not available')
device = "cuda" if torch.cuda.is_available() else "cpu"

_dataset_dir = "./food11"

batch_size = 64
num_class = 11

# The number of test transforms
n_epochs = 10

# set a random seed for reproducibility
myseed = 6666 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

p_tfm = 0.9
# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    transforms.RandomRotation(degrees=30, expand=True),

    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # You may add some transforms here.
    
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.02, hue=0.02),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
    transforms.RandomPosterize(bits=4, p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.2),
    transforms.RandomAutocontrast(p=0.3),
    # transforms.GaussianBlur(kernel_size=(1,3), sigma=(0.1, 1)),
    # transforms.RandomEqualize(p=0.2),

    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

no_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm, files=None, no_tfm=no_tfm, p=p_tfm):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
        self.no_transform = no_tfm
        self.p = p
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        if (random.uniform(0, 1) < self.p):
            im = self.transform(im)
        else:
            im = self.no_transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

resnet18 = models.resnet18(pretrained=False).to(device)
resnet34 = models.resnet34(pretrained=False).to(device)
resnet50 = models.resnet50(pretrained=False).to(device)

"""# Testing and generate prediction CSV"""

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm, no_tfm=no_tfm, p=p_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_transform_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm, no_tfm=no_tfm, p=1)
test_transform_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

resnet18 = models.resnet18(pretrained=False).to(device)
resnet34 = models.resnet34(pretrained=False).to(device)
resnet50 = models.resnet50(pretrained=False).to(device)

resnet18 = nn.DataParallel(resnet18)
resnet34 = nn.DataParallel(resnet34)
resnet50 = nn.DataParallel(resnet50)

resnet18.load_state_dict(torch.load(f"resnet_18_best.ckpt"))
resnet34.load_state_dict(torch.load(f"resnet_34_best.ckpt"))
resnet50.load_state_dict(torch.load(f"resnet_50_best.ckpt"))

resnet18.eval()
resnet34.eval()
resnet50.eval()

best_models = [resnet18, resnet34, resnet50]
weights = [1, 1, 1]

# prediction = []
# with torch.no_grad():
#     test_pbar = tqdm(test_loader, position=0, leave=True)
#     for data,_ in test_pbar:
#         test_pred = np.zeros((data.shape[0], 1000))
#         for i in range(len(best_models)):
#             model_best = best_models[i]
#             this_model_pred = model_best(data.to(device)).cpu().data.numpy()
#             # print(this_model_pred.shape)
#             test_pred = np.add(test_pred, this_model_pred * weights[i])
#         test_label = np.argmax(test_pred, axis=1)
#         prediction += test_label.squeeze().tolist()

prediction = []
with torch.no_grad():
    test_pbar = tqdm(test_loader, position=0, leave=True)
    for data,_ in test_pbar:
        test_pred = np.zeros((data.shape[0], 1000))
        for i in range(len(best_models)):
            model_best = best_models[i]
            this_model_pred = model_best(data.to(device)).cpu().data.numpy()
            # print(this_model_pred.shape)
            test_pred = np.add(test_pred, this_model_pred * weights[i])
        prediction += test_pred.tolist()

trans_prediction = np.zeros((len(test_transform_set), 1000))
with torch.no_grad():
    for epoch in range(n_epochs):
        test_pbar = tqdm(test_transform_loader, position=0, leave=True)
        pred_one_epoch = []
        for data,_ in test_pbar:
            test_pred = np.zeros((data.shape[0], 1000))
            for i in range(len(best_models)):
                model_best = best_models[i]
                this_model_pred = model_best(data.to(device)).cpu().data.numpy()
                # print(this_model_pred.shape)
                test_pred = np.add(test_pred, this_model_pred * weights[i])
            pred_one_epoch += test_pred.tolist()
        trans_prediction = np.add(trans_prediction, np.array(pred_one_epoch))

test_results = np.add(np.array(prediction) * n_epochs, trans_prediction)
prediction = np.argmax(test_results, axis=1)

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv(f'{_exp_name}.csv',index = False)