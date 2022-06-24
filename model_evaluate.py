#%%
from matplotlib import markers
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from net import VAE
from cvae_net import CVAE
# from VAE import load_image  
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from cVAE_train import load_image
from torchvision.utils import save_image
import torchvision.utils as vutils
from fourier_ring_corr import FSC,compute_frc

#%%
from cVAE_train import load_cryoET
from cvae_net_2 import CVAE
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = load_cryoET(batch_size,device)
cvae_net = CVAE(zsize=32, layer_count=5,channels=1,filter_base=32).to(device)
cvae_net.load_state_dict(torch.load('CVAEmodel_zdim32_ET5_unet.pkl')) # load trained model weight 
cvae_net.eval()
#%%
x_tilde_ls = []
img_ind = 4
(x,y) = next(iter(test_loader))
mu1,logvar1 = cvae_net.encode_y(y)

for i in range(50):
    z_sample = torch.randn(32, 32).to(device)
    z_sample = z_sample * torch.exp(0.5*logvar1) + mu1
    x_tilde = cvae_net.decode(z_sample,y)
    x_tilde_ls.append(x_tilde.detach().cpu())
x_tilde_batch_samples = torch.cat(x_tilde_ls,axis=1)
x_tilde_batch_mean = torch.mean(x_tilde_batch_samples,1)
save_image(vutils.make_grid(x_tilde_batch_mean[:,None], padding=5, normalize=True), 'cvae_ET_results/test_restore_ET_batch.png' , nrow=8)
save_image(vutils.make_grid(y, padding=5, normalize=True), 'cvae_ET_results/test_y_ET_batch.png' , nrow=8)
save_image(vutils.make_grid(x, padding=5, normalize=True), 'cvae_ET_results/test_x_ET_batch.png' , nrow=8)
#%%
x_tilde_ls = []
img_ind = 5
for i in range(50):
    z_sample = torch.randn(32, 32).to(device)
    z_sample = z_sample * torch.exp(0.5*logvar1) + mu1
    x_tilde = cvae_net.decode(z_sample,y)
    x_tilde_ls.append(x_tilde.detach().cpu()[img_ind:img_ind+1])
x_tilde_img1_samples = torch.cat(x_tilde_ls,axis=0)
save_image(vutils.make_grid(x_tilde_img1_samples[:32], padding=5, normalize=True), 'cvae_ET_results/test_one_restore_samples.png' , nrow=8)
# %%
x_orig = x[img_ind:img_ind+1].detach().cpu().numpy()[0,0]
y_orig = y[img_ind:img_ind+1].detach().cpu().numpy()[0,0]
x_hats = x_tilde_img1_samples.detach().cpu().numpy()[:,0]

for i in range(len(x_hats)):
    density,edge = compute_frc(x_hats[i],x_orig)#+np.random.normal(0,0.1,img1.shape))
    plt.plot(density,'b',alpha=0.2)
density,edge = compute_frc(y_orig,x_orig)#+np.random.normal(0,0.1,img1.shape))
plt.plot(density,'y-',alpha=1)
f = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.xticks(np.linspace(0,45,11),f)
plt.xlabel('Spacial Frequency')
plt.ylabel('FRC')
plt.savefig('frc_given_y.png',dpi=300)
plt.show()  
# %%
