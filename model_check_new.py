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
from cVAE_train_new import load_image
from torchvision.utils import save_image
import torchvision.utils as vutils
def rescaleUint8(img_array):
    # rescale 0~1 float array to 0~255 uint8 array
    img_array = (img_array - img_array.min(axis=(1,2),keepdims=True))/(img_array.max(axis=(1,2),keepdims=True)-img_array.min(axis=(1,2),keepdims=True)+1e-5)*255
    return img_array.astype(np.uint8)

#%%
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#vae_net = VAE(zsize=32, layer_count=5,channels=1,filter_base=128).to(device)
#vae_net.load_state_dict(torch.load('VAEmodel_zdim32_fb128.pkl')) # load trained model weight
#decoder = vae_net.decode
#z_sample = torch.randn(10, 32).to(device)
#x_tilde = decoder(z_sample)
#x_tilde = x_tilde.detach().cpu().numpy()

#%%
#x1 = x_tilde[0]
#x1_rescale = rescaleUint8(x1)
#display(Image.fromarray(x1_rescale[0]))
#plt.hist(x1.flatten(), bins=100)
#plt.show()
# %%
train_loader, test_loader = load_image(32,device)
#%%
#CVAE q(z|x,y) and q(z|y)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cvae_net = CVAE(zsize=32, layer_count=5,channels=1,filter_base=32).to(device)
cvae_net.load_state_dict(torch.load('CVAEmodel_zdim32_new.pkl')) # load trained model weight 
cvae_net.eval()
z0_ls = []
z1_ls = []
mu0_ls = []
mu1_ls = []
for i,(x,y) in enumerate(test_loader):
    if i >3:
        break
    mu0,logvar0 = cvae_net.encode_xy(x,y)
    z0 = mu0 + torch.exp(0.5*logvar0) * torch.randn_like(mu0)
    mu1,logvar1 = cvae_net.encode_y(y)
    z1 = mu1 + torch.exp(0.5*logvar1) * torch.randn_like(mu1)
    z0_ls.append(z0.detach().cpu().numpy())
    z1_ls.append(z1.detach().cpu().numpy())
    mu0_ls.append(mu0.detach().cpu().numpy())
    mu1_ls.append(mu1.detach().cpu().numpy())
z0 = np.concatenate(z0_ls,axis=0)
z1 = np.concatenate(z1_ls,axis=0)
mu0 = np.concatenate(mu0_ls,axis=0)
mu1 = np.concatenate(mu1_ls,axis=0)
#%%
z0_std = np.std(z0,axis=0,keepdims=True)
z1_std = np.std(z1,axis=0,keepdims=True)
z0_mean = np.mean(z0,axis=0,keepdims=True)
z1_mean = np.mean(z1,axis=0,keepdims=True)
# plt.plot(z0_std)
# plt.plot(z1_std)
# plt.show()
# index of largest two std
ind = np.argsort(z0_std)[0][-2:]

#%%
plt.plot(z0[:,ind[0]],z0[:,ind[1]],'ro',markersize=2)
plt.plot(z1[:,ind[0]],z1[:,ind[1]],'bo',markersize=2)
plt.plot(z0[0,ind[0]],z0[0,ind[1]],'r^',markersize=10)
plt.plot(z1[0,ind[0]],z1[0,ind[1]],'b^',markersize=10)
plt.plot(z0[1,ind[0]],z0[1,ind[1]],'rv',markersize=10)
plt.plot(z1[1,ind[0]],z1[1,ind[1]],'bv',markersize=10)
plt.plot(z0[2,ind[0]],z0[2,ind[1]],'rs',markersize=10)
plt.plot(z1[2,ind[0]],z1[2,ind[1]],'bs',markersize=10)
plt.show()
plt.save("./scatter1_new.png")

plt.plot(mu0[:,ind[0]],mu0[:,ind[1]],'ro',markersize=2)
plt.plot(mu1[:,ind[0]],mu1[:,ind[1]],'bo',markersize=2)
plt.plot(mu0[0,ind[0]],mu0[0,ind[1]],'r^',markersize=10)
plt.plot(mu1[0,ind[0]],mu1[0,ind[1]],'b^',markersize=10)
plt.plot(mu0[1,ind[0]],mu0[1,ind[1]],'rv',markersize=10)
plt.plot(mu1[1,ind[0]],mu1[1,ind[1]],'bv',markersize=10)
plt.plot(mu0[2,ind[0]],mu0[2,ind[1]],'rs',markersize=10)
plt.plot(mu1[2,ind[0]],mu1[2,ind[1]],'bs',markersize=10)
plt.show()
plt.save("./scatter2_new.png")
#%%

# %%
# draw 64 z from 32 dim z from normal distribution with mean and std of z1_mean and z1_std
x_tilde_ls = []
z_std = torch.from_numpy(z1_std).float().to(device) 
z_mean = torch.from_numpy(z1_mean).float().to(device)
(x,y) = next(iter(test_loader))
for i in range(16):
    z_sample = torch.randn(32, 32).to(device)
    z_sample = z_sample * z_std + z_mean
    x_tilde = cvae_net.decode(z_sample,y)
    x_tilde_ls.append(x_tilde.detach().cpu().numpy()[1:2])
x_tilde = np.concatenate(x_tilde_ls,axis=0)
x_tilde = torch.from_numpy(x_tilde).float()
save_image(vutils.make_grid(x_tilde[:16], padding=5, normalize=True), './samples_new.png' , nrow=8)

# %%
x_tilde_ls = []
img_ind = 1
(x,y) = next(iter(test_loader))
mu1,logvar1 = cvae_net.encode_y(y)

for i in range(16):
    z_sample = torch.randn(32, 32).to(device)
    z_sample = z_sample * torch.exp(0.5*logvar1) + mu1
    x_tilde = cvae_net.decode(z_sample,y)
    x_tilde_ls.append(x_tilde.detach().cpu().numpy()[img_ind:img_ind+1])
x_tilde = np.concatenate(x_tilde_ls,axis=0)
x_tilde = torch.from_numpy(x_tilde).float()
save_image(vutils.make_grid(x_tilde[:16], padding=5, normalize=True), './samples_given_y_new.png' , nrow=8)

# %%
