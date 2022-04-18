#%%
# !stylegan2_pytorch --data ./faces --name stylegan2 --models_dir ./models --results_dir ./results --image-size 64 --num-train-steps 100000

#%%
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = '.',
    name = 'stylegan2'
)

n_output = 1000
noise   = torch.randn(n_output, 512)
styles  = loader.noise_to_styles(noise, trunc_psi = 0.75)

imgs_sample = loader.styles_to_images(styles[:32])
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

#%%
os.makedirs('./stylegan_output', exist_ok=True)
eval_batch_size = 100
for i in range(10):
    images = loader.styles_to_images(styles[i*eval_batch_size:(i+1)*eval_batch_size])
    for j in range(eval_batch_size):
        save_image(images[j], f'./stylegan_output/{i*eval_batch_size+j+1}.jpg')

#%%
# %cd stylegan_output
# !tar -zcf ../stylegan2.tgz ./*.jpg
# %cd ..

#%%
