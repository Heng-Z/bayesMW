# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#%%
from __future__ import print_function
from asyncio import base_tasks
import torch.utils.data
# from scipy import misc
from torch import diag_embed, optim
from torchvision.utils import save_image
import torchvision.utils as vutils
from cvae_net import *
import numpy as np
import pickle
import time
import random
import os
# from dlutils import batch_provider
# from dlutils.pytorch.cuda_helper import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import shutil
import matplotlib.pyplot as plt

im_size = 128


def loss_function(recon_x, x, mu0, logvar0, mu1, logvar1):
    recon_x_flat = recon_x.view(len(recon_x), -1)
    x_flat = x.view(len(x), -1)
    BCE = torch.mean(torch.mean(0.5*(recon_x_flat - x_flat)**2,1))
    KLD = 0.5*torch.mean(torch.mean(torch.exp(logvar0)/torch.exp(logvar1) + (mu0-mu1)**2/(torch.exp(logvar1)) - 1 - logvar0+logvar1, 1))
    # TODO:
    # add the KL divergence loss between teacher encoder ouput and standard normal
    return BCE, KLD

def load_image(bs,device):
    ffhq_path = './ffhq_images_grey/'
    ffhq_mw_path = './ffhq_images_grey_mw/'
    # load every image in the folder and organize them in a tensor
    X_list = os.listdir(ffhq_path)
    X_list.sort()
    Y_list = os.listdir(ffhq_mw_path)
    Y_list.sort()
    train_X_list = X_list[:int(len(X_list)*0.9)]
    train_Y_list = Y_list[:int(len(Y_list)*0.9)]
    test_X_list = X_list[int(len(X_list)*0.9):]
    test_Y_list = Y_list[int(len(Y_list)*0.9):]
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []
    for img in train_X_list:
        train_X.append(np.asarray(Image.open(ffhq_path+img)))
    for img in train_Y_list:
        train_Y.append(np.asarray(Image.open(ffhq_mw_path+img)))
    for img in test_X_list:
        test_X.append(np.asarray(Image.open(ffhq_path+img)))
    for img in test_Y_list:
        test_Y.append(np.asarray(Image.open(ffhq_mw_path+img)))
    # preprocess the images
    train_X = preprocess(np.array(train_X))
    train_Y = preprocess(np.array(train_Y))
    test_X = preprocess(np.array(test_X))
    test_Y = preprocess(np.array(test_Y))
    # create the tensor
    train_X = torch.from_numpy(train_X[:,None,:,:]).float().to(device)
    train_Y = torch.from_numpy(train_Y[:,None,:,:]).float().to(device)
    test_X = torch.from_numpy(test_X[:,None,:,:]).float().to(device)
    test_Y = torch.from_numpy(test_Y[:,None,:,:]).float().to(device)

    # get my dataloader
    train_loader = DataLoader(TensorDataset(train_X,train_Y), batch_size=bs, shuffle=False)

    test_loader = DataLoader(TensorDataset(test_X,test_Y), batch_size=bs, shuffle=False)
    return train_loader, test_loader

def preprocess(img_array):
    # image_array is of shape (total_len,  height, width)
    # normalize the image array
    # print(img_array.shape)
    img_array = img_array.astype(np.float32)
    img_array = (img_array - np.percentile(img_array,3,axis=[1,2],keepdims=True)) / (np.percentile(img_array,97,axis=[1,2],keepdims=True)-np.percentile(img_array,3,axis=[1,2],keepdims=True)+1e-5)
    return img_array

def main():
    directory = './cvae_result_test3'
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copy('./cVAE_train.py', directory)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    z_size = 32
    filter_base = 32
    batch_norm = True
    lr_decay = 200
    kl_factor = 0.1
    cvae = CVAE(zsize=z_size, layer_count=5,channels=1,filter_base=filter_base,batch_norm=batch_norm).to(device)
    cvae.train()
    cvae.weight_init(mean=0, std=0.02)

    lr = 0.00005
    vae_optimizer = optim.Adam(cvae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch = 500
    train_loader, test_loader = load_image(batch_size,device)
    # test_loader,train_loader = load_image(batch_size,device)
    for epoch in range(train_epoch):
        cvae.train()

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()
        if (epoch + 1) % 100 == 0:
            kl_factor *= 1
            print("kl_factor: ", kl_factor)
        if (epoch + 1) % lr_decay == 0:
            vae_optimizer.param_groups[0]['lr'] *= 0.5
            print("learning rate change!")

   
        for i,(x,y) in enumerate(train_loader):
            cvae.train()
            cvae.zero_grad()
            rec, mu0, logvar0, mu1, logvar1= cvae(x,y)

            loss_re, loss_kl = loss_function(rec, x, mu0, logvar0, mu1, logvar1)
            (loss_re + kl_factor*loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################

            # os.makedirs('results_rec', exist_ok=True)
            # os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 10
            if (i+1) % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
        #         with torch.no_grad():
        #             vae.eval()
        #             x_rec, _, _ = vae(x)
        #             resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
        #             resultsample = resultsample.cpu()
        #             save_image(resultsample.view(-1, 3, im_size, im_size),
        #                        'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
        #             x_rec = vae.decode(sample1)
        #             resultsample = x_rec * 0.5 + 0.5
        #             resultsample = resultsample.cpu()
        #             save_image(resultsample.view(-1, 3, im_size, im_size),
        #                        'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        # del batches
        # del data_train
        
        
        if (epoch+1)%50==0:
            for j, (x,y) in enumerate(test_loader):
                cvae.eval()

                out = x.data.cpu()
                # out = (out + 1) / 2
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/x%s.png' % (epoch+1), nrow=8)
                out = y.data.cpu()
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/y%s.png' % (epoch+1), nrow=8)
                out = cvae.decode(cvae.encode_xy(x,y)[0],y)  #out=x_tilde
                out = out.data.cpu()
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/xyrecon%s.png' % (epoch+1), nrow=8)
                out = cvae.decode(cvae.encode_y(y)[0],y)  #out=x_tilde
                out = out.data.cpu()
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/yrecon%s.png' % (epoch+1), nrow=8)
                
                # out = cvae.decode(torch.randn([64,z_size]).to(device))  ##out=x_p
                # out = out.data.cpu()
                # # out = (out + 1) / 2
                # save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/sample%s.png' % (epoch+1), nrow=8)
                break
            
            # z_list = []
            # for j, x in enumerate(test_loader):
            #     if type(x) is list:
            #         x = x[0]
            #     mu, logvar = vae.encode(x)
            #     mu = mu.squeeze()
            #     logvar = logvar.squeeze()
            #     z = vae.reparameterize(mu, logvar)
            #     z_list.append(z.data.cpu().numpy())
            # z_cat = np.concatenate(z_list, axis=0)


            # from scipy.stats import norm

            # s = np.linspace(-3, 3, 50)

            # fig = plt.figure(figsize=(20, 20))
            # fig.subplots_adjust(hspace=0.6, wspace=0.4)

            # for i in range(30):
            #     ax = fig.add_subplot(3, 10, i+1)
            #     ax.hist(z_cat[:,i], density=True, bins = 50)
            #     ax.axis('off')
            #     ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
            #     ax.plot(s,norm.pdf(s))

            # plt.savefig(directory+'/hist%s.png' % (epoch+1))

    print("Training finish!... save training results")
    torch.save(cvae.state_dict(), "CVAEmodel_zdim32.pkl")

if __name__ == '__main__':
    main()

