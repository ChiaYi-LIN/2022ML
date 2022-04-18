#%%
"""## Other setting"""

# import module
import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm


# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2022)
workspace_dir = '.'

#%%
"""# Dataset preparation
In this section, we prepare for the dataset for Pytorch

## Create dataset for Pytorch

In order to unified image information, we use the transform function to:
1. Resize image to 64x64
2. Normalize the image

This CrypkoDataset class will be use in Section 4
"""

# prepare for CrypkoDataset

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

#%%
"""## Show the image
Show some sample in the dataset
"""

temp_dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

images = [temp_dataset[i] for i in range(4)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

#%%
"""# Model setting
In this section, we will create models and trainer.

## Create model
In this section, we will create models for Generator and Discriminator
"""

# Generator

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self, in_dim, feature_dim=64):
        super().__init__()
    
        #input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),               #(batch, feature_dim * 16, 8, 8)     
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),               #(batch, feature_dim * 16, 16, 16)     
            self.dconv_bn_relu(feature_dim * 2, feature_dim),                   #(batch, feature_dim * 16, 32, 32)     
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()   
        )
        self.apply(weights_init)
    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),        #double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator
class Discriminator_WGAN(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator_WGAN, self).__init__()
            
        #input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:

        Remove last sigmoid layer for WGAN
        """
        # DCGAN
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   #(batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               #(batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               #(batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() 
        )

        # WGAN
        self.l1_wgan = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
        )
        self.l2_wgan = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1), #(batch, 3, 16, 16)
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
        ) 
        self.l3_wgan = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1), #(batch, 3, 8, 8)
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
        ) 
        self.l4_wgan = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1), #(batch, 3, 4, 4)
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),
        ) 
        self.out_wgan = nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)

        # WGAN-GP
        self.l1_wgan_gp = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
        )
        self.l2_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1), #(batch, 3, 16, 16)
            nn.InstanceNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
        ) 
        self.l3_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1), #(batch, 3, 8, 8)
            nn.InstanceNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
        ) 
        self.l4_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1), #(batch, 3, 4, 4)
            nn.InstanceNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),
        ) 
        self.out_wgan_gp = nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)

        self.apply(weights_init)
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        NOTE FOR SETTING DISCRIMINATOR:

        You can't use nn.Batchnorm for WGAN-GP
        Use nn.InstanceNorm2d instead
        """

        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        # DCGAN
        # y = self.l1(x)
        # y = y.view(-1)

        # WGAN
        out_l1 = self.l1_wgan(x)
        out_l2 = self.l2_wgan(out_l1)
        out_l3 = self.l3_wgan(out_l2)
        out_l4 = self.l4_wgan(out_l3)
        y = self.out_wgan(out_l4).view(-1)

        # WGAN-GP
        # out_l1 = self.l1_wgan_gp(x)
        # out_l2 = self.l2_wgan_gp(out_l1)
        # out_l3 = self.l3_wgan_gp(out_l2)
        # out_l4 = self.l4_wgan_gp(out_l3)
        # y = self.out_wgan_gp(out_l4).view(-1)
        return y

class Discriminator_WGAN_GP(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator_WGAN_GP, self).__init__()
            
        #input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:

        Remove last sigmoid layer for WGAN
        """
        # DCGAN
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   #(batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               #(batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               #(batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() 
        )

        # WGAN
        self.l1_wgan = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
        )
        self.l2_wgan = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1), #(batch, 3, 16, 16)
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
        ) 
        self.l3_wgan = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1), #(batch, 3, 8, 8)
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
        ) 
        self.l4_wgan = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1), #(batch, 3, 4, 4)
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),
        ) 
        self.out_wgan = nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)

        # WGAN-GP
        self.l1_wgan_gp = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
        )
        self.l2_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1), #(batch, 3, 16, 16)
            nn.InstanceNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
        ) 
        self.l3_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1), #(batch, 3, 8, 8)
            nn.InstanceNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
        ) 
        self.l4_wgan_gp = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1), #(batch, 3, 4, 4)
            nn.InstanceNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),
        ) 
        self.out_wgan_gp = nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)

        self.apply(weights_init)
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        NOTE FOR SETTING DISCRIMINATOR:

        You can't use nn.Batchnorm for WGAN-GP
        Use nn.InstanceNorm2d instead
        """

        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        # DCGAN
        # y = self.l1(x)
        # y = y.view(-1)

        # WGAN
        # out_l1 = self.l1_wgan(x)
        # out_l2 = self.l2_wgan(out_l1)
        # out_l3 = self.l3_wgan(out_l2)
        # out_l4 = self.l4_wgan(out_l3)
        # y = self.out_wgan(out_l4).view(-1)

        # WGAN-GP
        out_l1 = self.l1_wgan_gp(x)
        out_l2 = self.l2_wgan_gp(out_l1)
        out_l3 = self.l3_wgan_gp(out_l2)
        out_l4 = self.l4_wgan_gp(out_l3)
        y = self.out_wgan_gp(out_l4).view(-1)
        return y

# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#%%
"""## Create trainer
In this section, we will create a trainer which contains following functions:
1. prepare_environment: prepare the overall environment, construct the models, create directory for the log and ckpt
2. train: train for generator and discriminator, you can try to modify the code here to construct WGAN or WGAN-GP
3. inference: after training, you can pass the generator ckpt path into it and the function will save the result for you
"""

class TrainerGAN_WGAN():
    def __init__(self, config):
        self.config = config
        
        self.G = Generator(100)
        self.D = Discriminator_WGAN(3)
        
        self.loss = nn.BCELoss()

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        # GAN & WGAN-GP
        # self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        # self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        # WGAN
        self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=self.config["lr"])
        self.opt_G = torch.optim.RMSprop(self.G.parameters(), lr=self.config["lr"])
        
        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()
        
    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # update dir by time
        # time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, f'{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, f'{self.config["model_type"]}')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # model preparation
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()
    def gp(self, real_samples, fake_samples):
        """
        Implement gradient penalty function
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
        # print(f'alpha.shape = {alpha.shape}')
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print(f'interpolates.shape = {interpolates.shape}')
        d_interpolates = self.D(interpolates)
        # print(f'd_interpolates.shape = {d_interpolates.shape}')
        fake = Variable(torch.Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False).cuda()
        # print(f'fake.shape = {fake.shape}')
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()


                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                NOTE FOR SETTING DISCRIMINATOR LOSS:
                
                GAN: 
                    loss_D = (r_loss + f_loss)/2
                WGAN: 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: 
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """
                # Loss for discriminator
                # DCGAN
                # r_loss = self.loss(r_logit, r_label)
                # f_loss = self.loss(f_logit, f_label)
                # loss_D = (r_loss + f_loss) / 2

                # WGAN
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

                # WGAN-GP
                # gradient_penalty = self.gp(r_imgs, f_imgs)
                # loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + config_WGAN["lambda_gp"] * gradient_penalty

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                NOTE FOR SETTING WEIGHT CLIP:
                
                WGAN: below code
                """
                # WGAN
                # DCGAN & WGAN-GP don't do this
                for p in self.D.parameters():
                    p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])



                # *********************
                # *    Train G        *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)


                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # DCGAN
                    # Loss for the generator.
                    # loss_G = self.loss(f_logit, r_label)

                    # WGAN & WGAN-GP
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.G.train()

            if (e+1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

        logging.info('Finish training')

    def inference(self, G_path, n_generate=1000, n_output=30, show=False):
        """
        1. G_path is the path for Generator ckpt
        2. You can use this function to generate final answer
        """

        self.G.load_state_dict(torch.load(G_path))
        self.G.cuda()
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).cuda()
        imgs = (self.G(z).data + 1) / 2.0
        
        os.makedirs('output', exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i+1}.jpg')
        
        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

class TrainerGAN_WGAN_GP():
    def __init__(self, config):
        self.config = config
        
        self.G = Generator(100)
        self.D = Discriminator_WGAN_GP(3)
        
        self.loss = nn.BCELoss()

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        # GAN & WGAN-GP
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        # WGAN
        # self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=self.config["lr"])
        # self.opt_G = torch.optim.RMSprop(self.G.parameters(), lr=self.config["lr"])
        
        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()
        
    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # update dir by time
        # time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, f'{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, f'{self.config["model_type"]}')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # model preparation
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()
    def gp(self, real_samples, fake_samples):
        """
        Implement gradient penalty function
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
        # print(f'alpha.shape = {alpha.shape}')
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print(f'interpolates.shape = {interpolates.shape}')
        d_interpolates = self.D(interpolates)
        # print(f'd_interpolates.shape = {d_interpolates.shape}')
        fake = Variable(torch.Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False).cuda()
        # print(f'fake.shape = {fake.shape}')
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()


                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                NOTE FOR SETTING DISCRIMINATOR LOSS:
                
                GAN: 
                    loss_D = (r_loss + f_loss)/2
                WGAN: 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: 
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """
                # Loss for discriminator
                # DCGAN
                # r_loss = self.loss(r_logit, r_label)
                # f_loss = self.loss(f_logit, f_label)
                # loss_D = (r_loss + f_loss) / 2

                # WGAN
                # loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

                # WGAN-GP
                gradient_penalty = self.gp(r_imgs, f_imgs)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + config_WGAN_GP["lambda_gp"] * gradient_penalty

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                NOTE FOR SETTING WEIGHT CLIP:
                
                WGAN: below code
                """
                # WGAN
                # DCGAN & WGAN-GP don't do this
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])



                # *********************
                # *    Train G        *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)


                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # DCGAN
                    # Loss for the generator.
                    # loss_G = self.loss(f_logit, r_label)

                    # WGAN & WGAN-GP
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.G.train()

            if (e+1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

        logging.info('Finish training')

    def inference(self, G_path, n_generate=1000, n_output=30, show=False):
        """
        1. G_path is the path for Generator ckpt
        2. You can use this function to generate final answer
        """

        self.G.load_state_dict(torch.load(G_path))
        self.G.cuda()
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).cuda()
        imgs = (self.G(z).data + 1) / 2.0
        
        os.makedirs('output', exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i+1}.jpg')
        
        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

#%%
config_WGAN = {
    "model_type": "WGAN",
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 3,
    "n_critic": 1,
    "z_dim": 100,
    "workspace_dir": workspace_dir, # define in the environment setting
    "clip_value": 1.0,
    "lambda_gp": 10,
}

config_WGAN_GP = {
    "model_type": "WGAN-GP",
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 3,
    "n_critic": 1,
    "z_dim": 100,
    "workspace_dir": workspace_dir, # define in the environment setting
    "clip_value": 1.0,
    "lambda_gp": 10,
}

#%%
trainer_WGAN = TrainerGAN_WGAN(config_WGAN)

#%%
trainer_WGAN.train()

#%%
trainer_WGAN_GP = TrainerGAN_WGAN_GP(config_WGAN_GP)

#%%
trainer_WGAN_GP.train()

#%%
for i, data in enumerate(trainer_WGAN.dataloader):
    imgs = data.cuda()
    bs = imgs.size(0)

    z = Variable(torch.randn(bs, trainer_WGAN.config["z_dim"])).cuda()
    r_imgs = Variable(imgs).cuda()
    f_imgs = trainer_WGAN.G(z)

    r_logit = trainer_WGAN.D(r_imgs)
    f_logit = trainer_WGAN.D(f_imgs)

    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

    trainer_WGAN.D.zero_grad()
    loss_D.backward()

    grad_norm_WGAN = []
    for layer in [trainer_WGAN.D.l4_wgan[0], trainer_WGAN.D.l3_wgan[0], trainer_WGAN.D.l2_wgan[0], trainer_WGAN.D.l1_wgan[0]]:
        total_norm = 0
        for p in layer.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        grad_norm_WGAN.append(np.log(total_norm))
    
    break


#%%
grad_norm_WGAN

#%%
for i, data in enumerate(trainer_WGAN_GP.dataloader):
    imgs = data.cuda()
    bs = imgs.size(0)

    z = Variable(torch.randn(bs, trainer_WGAN_GP.config["z_dim"])).cuda()
    r_imgs = Variable(imgs).cuda()
    f_imgs = trainer_WGAN_GP.G(z)

    r_logit = trainer_WGAN_GP.D(r_imgs)
    f_logit = trainer_WGAN_GP.D(f_imgs)

    gradient_penalty = trainer_WGAN_GP.gp(r_imgs, f_imgs)
    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + config_WGAN_GP["lambda_gp"] * gradient_penalty

    trainer_WGAN_GP.D.zero_grad()
    loss_D.backward()

    grad_norm_WGAN_GP = []
    for layer in [trainer_WGAN_GP.D.l4_wgan_gp[0], trainer_WGAN_GP.D.l3_wgan_gp[0], trainer_WGAN_GP.D.l2_wgan_gp[0], trainer_WGAN_GP.D.l1_wgan_gp[0]]:
        total_norm = 0
        for p in layer.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        grad_norm_WGAN_GP.append(np.log(total_norm))
    
    break

#%%
grad_norm_WGAN_GP

#%%
import matplotlib.pyplot as plt
plt.plot([4, 3, 2, 1], grad_norm_WGAN, label = f"WGAN c={config_WGAN['clip_value']}")
plt.plot([4, 3, 2, 1], grad_norm_WGAN_GP, label = "WGAN-GP")

# plt.xlim(4, 1)
plt.ylim(-20, 20)
plt.xlabel('Discriminator Layer')
plt.ylabel('Gradient Norm (log scale)')
plt.title('Gradient Norm of Discriminator Layers of WGAN and WGAN-GP')
plt.legend()
plt.savefig('./output.jpg')
# plt.show()

#%%
