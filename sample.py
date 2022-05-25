#%%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from net import VAE
from VAE import load_image  
from torch.autograd import Variable
import matplotlib.pyplot as plt

def apply_mw(x,mw):
    # Simulate missing wedge effect
    # x: np array of images; shape: (batch_size, 1, im_size, im_size)
    # return: np array of degraded images; shape: (batch_size, 1, im_size, im_size); with missing-wedge effect
    xdim = x.shape[2]
    ydim = x.shape[3]
    outData = torch.zeros_like(x).to(x.device)
    for i, item in enumerate(x):
        img_one = item[0]
        # A*x : simulate missing wedge effect
        outData_i=torch.fft.ifft2(torch.fft.fftshift(mw) * torch.fft.fft2(img_one))
        outData[i] = torch.real(outData_i)[None,:,:]
    return outData

# missing wedge filter
class TwoDPsf:
    def __init__(self,size_y,size_x):
        self._dimension=(size_y,size_x)

    def getMW(self,missingAngle=[30,30]):
        self._mw=np.zeros((self._dimension[0],self._dimension[1]),dtype=np.double)
        missingAngle = np.array(missingAngle)
        missing=np.pi/180*(90-missingAngle) * self._dimension[0]/self._dimension[1]
        for i in range(self._dimension[0]):
            for j in range(self._dimension[1]):
                y=(i-self._dimension[0]/2)
                x=(j-self._dimension[1]/2)
                if x==0:# and y!=0:
                    theta=np.pi/2
                #elif x==0 and y==0:
                #    theta=0
                #elif x!=0 and y==0:
                #    theta=np.pi/2
                else:
                    theta=abs(np.arctan(y/x))

                if 4*x**2/self._dimension[1]**2 + 4*y**2 / self._dimension[0]**2  <= 1:
                    if x > 0 and y > 0 and theta < missing[0]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x < 0 and y < 0 and theta < missing[0]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x > 0 and y < 0 and theta < missing[1]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x < 0 and y > 0 and theta < missing[1]:
                        self._mw[i,j]=1#np.cos(theta)

                if int(y) == 0:
                    self._mw[i,j]=1
        return self._mw

# Loss and gradient for a pair of y and latent variable z
def loss_grad(y,z,decoder,sigma=1,grad=True):
    # y,z,decoder should be torch tensors within the same device(cpu or cuda)
    # decoder parameter requires_grad=False
    bs = y.shape[0]
    gradient = torch.zeros_like(z)
    losses = torch.zeros([bs,1])
    xdim,ydim = y.shape[2:]
    mw = TwoDPsf(xdim, ydim).getMW()
    mw = torch.from_numpy(mw).float().to(y.device)
    for i, zi in enumerate(z):
        zi =  Variable(zi[None,:],requires_grad=True)
        x_tilde = decoder(zi)
        y_tilde = apply_mw(x_tilde,mw)
        loss = torch.sum(0.5*(y_tilde-y)**2/sigma**2)
        losses[i] = loss
        if grad:
            loss.backward()
            gradient[i] = zi.grad
    if grad:
        return losses, gradient
    else:
        return losses






# %% Test gradient
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_net = VAE(zsize=128, layer_count=5,channels=1,filter_base=128).to(device)
vae_net.load_state_dict(torch.load('VAEmodel.pkl'))
decoder = vae_net.decode
for param in vae_net.parameters():
    param.requires_grad = False

y = torch.randn(2, 1, 128, 128).to(device)
z = torch.randn(2, 128).to(device)
losses, grad = loss_grad(y,z,decoder,sigma=1,grad=True)
print(grad)

# %% Test sampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_sample = torch.randn(100, 128).to(device)
x_tilde = decoder(z_sample)
x_tilde = x_tilde.detach().cpu().numpy()
plt.imshow(x_tilde[0,0,:,:])
plt.show()

# %%
