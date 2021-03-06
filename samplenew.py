#%%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from net import VAE
from VAE import load_image  
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

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
def loss_grad(y,z,decoder,mw,sigma=1,grad=True):
    # y,z,decoder should be torch tensors within the same device(cpu or cuda)
    # decoder parameter requires_grad=False
    assert y.device == z.device
    bs = y.shape[0]
    gradient = torch.zeros_like(z)
    losses = torch.zeros([bs,1]).to(y.device)
    xdim,ydim = y.shape[2:]
    for i, zi in enumerate(z):
        zi =  Variable(zi[None,:],requires_grad=True)
        t1 = time.time()
        x_tilde = decoder(zi)
        t2 = time.time()
        y_tilde = apply_mw(x_tilde,mw)
        #p(z|y) = exp(-||y-y_tilde||^2/2sigma_s^2)*exp(-||z-zi||^2/2)
        # loss = -log(p(z|y)) 
        loss = torch.sum(0.5*(y_tilde-y)**2/sigma**2)+torch.sum (0.5*zi**2)
        losses[i] = loss.data
        if grad:
            loss.backward()
            gradient[i] = zi.grad
        t3 = time.time()
        print('decoder: %.6f,bp: %.6f'%(t2-t1,t3-t1))
    if grad:
        return losses, gradient
    else:
        return losses


def preprocess(img_array):
    # image_array is of shape (total_len,  height, width)
    # normalize the image array
    # print(img_array.shape)
    img_array = img_array.astype(np.float32)
    img_array = (img_array - np.percentile(img_array,5,axis=[1,2],keepdims=True)) / (np.percentile(img_array,95,axis=[1,2],keepdims=True)-np.percentile(img_array,5,axis=[1,2],keepdims=True)+1e-5)
    return img_array

def rescaleUint8(img_array):
    # rescale 0~1 float array to 0~255 uint8 array
    img_array = (img_array - img_array.min(axis=(1,2),keepdims=True))/(img_array.max(axis=(1,2),keepdims=True)-img_array.min(axis=(1,2),keepdims=True)+1e-5)*255
    return img_array.astype(np.uint8)


# %% Test gradient
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_net = VAE(zsize=32, layer_count=5,channels=1,filter_base=32).to(device)
vae_net.load_state_dict(torch.load('VAEmodel_zdim32_v2.pkl')) # load trained model weight
vae_net.eval()
decoder = vae_net.decode # get decoder
for param in vae_net.parameters(): 
    param.requires_grad = False

y = torch.randn(1, 1, 128, 128).to(device) #(batch_size, channel, im_size, im_size) # load(image)
ffhq_path = './ffhq_images_grey_mw/'
img_list = os.listdir(ffhq_path)
test_images_list = img_list[int(len(img_list)*0.9):]
image_path=ffhq_path+test_images_list[1]
y_np = np.asarray(Image.open(image_path))
y_np = preprocess(np.array([y_np]))
y = torch.from_numpy(y_np[None,:,:,:]).to(device)
z = torch.randn(1, 32).to(device) # (batch_size, z_size)
losses, grad = loss_grad(y,z,decoder,mw,sigma=1,grad=True) # get loss and gradient of -log(p(z|y)) at z
#z = z + grad*eta + torch.randn_like(z)*0.1 # update z
#print(grad)

# %% Test sampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_sample = torch.randn(1, 32).to(device)
x_tilde = decoder(z_sample)
x_samples=[]
z_traj = []
#x_tilde = x_tilde.detach().cpu().numpy()
#plt.imshow(x_tilde[0,0,:,:])
#plt.show()
#HMC sampling
#define maximum step
Lmax=1000
epsilons=1e-2
sims=100
count=0
mw = TwoDPsf(128, 128).getMW()
mw = torch.from_numpy(mw).float().to(device)
for i in range(sims):
    r0=torch.randn_like(z_sample)
    h0=loss_grad(y,z_sample,decoder,mw,sigma=1,grad=False)
    rnew = r0.clone()
    lstep=torch.randint(1,Lmax,(1,)).to(device)
    for j in range(lstep):
        t1 = time.time()
        _, grad = loss_grad(y,z_sample,decoder,mw,sigma=1,grad=True)
        t2 = time.time()
        r12=rnew-(epsilons/2)*grad
        #thetanew=np.array([theta10,theta20])+epsilon*r12
        znew=(z_sample+epsilons*r12).float()
        _, grad = loss_grad(y,znew,decoder,mw,sigma=1,grad=True)
        rnew=r12-(epsilons/2)*grad
        t3 = time.time()
        z_sample=znew
        if j%10==0:
            print('grad time: %.5f one step time %.5f'%(t2-t1,t3-t1))
    
    h1=loss_grad(y,z_sample,decoder,mw,sigma=1,grad=False)
    a=min(1,torch.exp(h0-h1 + 0.5*torch.sum(r0**2)- 0.5*torch.sum(rnew**2)))
    print(a,lstep)
    if a>torch.rand(1).to(device):
        #theta1ran[i]=thetanew[0]
        #theta2ran[i]=thetanew[1]
        x_tilde = decoder(z_sample)
        x_samples.append(x_tilde)
        z_traj.append(z_sample)
        count+=1
    else:
        continue
        #theta1ran[i]=theta1ran[i-1]
        #theta2ran[i]=theta2ran[i-1]


# x_aver=(torch.sum(x_samples,0)/1000).to(device)
# print(x_aver.shape)
# x_aver = x_aver.detach().cpu().numpy()
# plt.imshow(x_aver[0,0,:,:])
# plt.show()
# %%
x_sample1 = x_samples[-1].detach().cpu().numpy()
x_sample1_rescale = rescaleUint8(x_sample1[0])
display(Image.fromarray(x_sample1_rescale[0]))
a = Image.fromarray(x_sample1_rescale[0])
a.save('lastface.png')
acorrlist=[]
z_trajnew=[]
x_samplenew=[]
for i in range(count):
    z_traj[i] = z_traj[i].detach().cpu().numpy()
    z_trajnew.append(z_traj[i])
z_trajnew=np.array(z_trajnew)

for i in range(32):
    ndata=z_trajnew[:,0,i]
    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
    acorrlist.append(acorr)
    plt.plot(np.arange(count),np.array(acorr))
    plt.show()
    plt.savefig('zcorrelation'+str(i)+'.png')
#acorr = numpy.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
# a.save('xxx')
# %%