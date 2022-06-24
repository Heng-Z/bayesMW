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

from __future__ import print_function
from asyncio import base_tasks
import torch.utils.data
# from scipy import misc
from torch import diag_embed, optim
from torchvision.utils import save_image
import torchvision.utils as vutils
from net import *
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


def loss_function(recon_x, x, mu, logvar,disc_recon,disc_orig):
    recon_x_flat = recon_x.view(len(recon_x), -1)
    x_flat = x.view(len(x), -1)
    # BCE = torch.mean(torch.sum(0.5*(recon_x_flat - x_flat)**2,1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    BCE = torch.mean(0.5*(recon_x_flat - x_flat)**2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    
    # Discriminator classification loss
    DC_disc = -torch.mean(torch.log(1-disc_recon+1e-5))
    DC_orig = - torch.mean(torch.log(disc_orig+1e-5))
    return BCE, 0.1*KLD,DC_disc,DC_orig


# def process_batch(batch):
#     data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

#     x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
#     x = x.view(-1, 3, im_size, im_size)
#     return x

def load_image(bs,device):
    ffhq_path = './ffhq_images_grey/'
    # load every image in the folder and organize them in a tensor
    img_list = os.listdir(ffhq_path)
    random.shuffle(img_list)
    train_images_list = img_list[:int(len(img_list)*0.9)]
    test_images_list = img_list[int(len(img_list)*0.9):]
    train_images = []
    test_images = []
    for img in train_images_list:
        train_images.append(np.asarray(Image.open(ffhq_path+img)))
    for img in test_images_list:
        test_images.append(np.asarray(Image.open(ffhq_path+img)))
    train_images = preprocess(np.array(train_images))
    test_images = preprocess(np.array(test_images))
    train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
    test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])
    train_images = torch.from_numpy(train_images).float().to(device)
    test_images = torch.from_numpy(test_images).float().to(device)

    # get my dataloader
    train_dataset = TensorDataset(train_images)
    test_dataset = TensorDataset(test_images)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    return train_loader, test_loader

def preprocess(img_array):
    # image_array is of shape (total_len,  height, width)
    # normalize the image array
    # print(img_array.shape)
    img_array = img_array.astype(np.float32)
    img_array = (img_array - np.percentile(img_array,5,axis=[1,2],keepdims=True)) / (np.percentile(img_array,95,axis=[1,2],keepdims=True)-np.percentile(img_array,5,axis=[1,2],keepdims=True)+1e-5)
    return img_array

def main():
    directory = './vae_gan_results_meanloss'
    model_name = './vae_gan_imfc200_fb32.pkl'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # shutil.copy('./VAE.py', directory)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    z_size = 32
    filter_base = 32
    batch_norm = True
    lr_decay = 400
    kl_factor = 0.1
    imitate_factor = 0.01
    imit_grow = 200
    vaegan = VAEGAN(zsize=z_size, layer_count=5,channels=1,filter_base=filter_base,batch_norm=batch_norm).to(device)
    vaegan.train()
    vaegan.weight_init(mean=0, std=0.02)

    lr = 0.0005
    m = 20
    vae_optimizer = optim.Adam(vaegan.vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    dic_optimizer = optim.Adam(vaegan.discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch = 2000
    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)
    train_loader, test_loader = load_image(batch_size,device)
    # test_loader,train_loader = load_image(batch_size,device)
    train_disc = True
    train_vae = True
    for epoch in range(train_epoch):
        vaegan.train()

        rec_loss = 0
        kl_loss = 0
        d_recon_loss = 0
        d_orig_loss = 0
        epoch_start_time = time.time()
        if (epoch + 1) % imit_grow == 0:
            imitate_factor = imitate_factor * 1.5 if imitate_factor < 999 else imitate_factor
            print('imitation factor:', imitate_factor)
        if (epoch + 1) % lr_decay == 0:
            vae_optimizer.param_groups[0]['lr'] /= 2
            dic_optimizer.param_groups[0]['lr'] /= 2
            
            print("learning rate change!")

   
        for i,x in enumerate(train_loader):
            if type(x) is list:
                x = x[0]

            vaegan.train()
            vaegan.zero_grad()
            x_tilde, mu, logvar,disc_recon, disc_orig = vaegan(x)

            loss_re, loss_kl, loss_disc_recon,loss_disc_orig = loss_function(x_tilde, x, mu, logvar,disc_recon, disc_orig)
            loss_gen = loss_re + kl_factor * loss_kl - imitate_factor*(loss_disc_recon)
            loss_disc = loss_disc_recon + loss_disc_orig
            # if True:
            if train_vae:
                loss_gen.backward(retain_graph=True)
                vae_optimizer.step()
                vaegan.zero_grad()
            if train_disc:
                loss_disc.backward(retain_graph=True)
                dic_optimizer.step()
                # if train_disc:
                #     # print the gradients
                #     print(vaegan.discriminator.conv1.weight.grad[0])
                vaegan.zero_grad()

            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()
            d_recon_loss += loss_disc_recon.item()
            d_orig_loss += loss_disc_orig.item()

            #############################################

            # os.makedirs('results_rec', exist_ok=True)
            # os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            if (i+1) % m == 0:
                rec_loss /= m
                kl_loss /= m
                d_recon_loss /= m
                d_orig_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.5f, KL loss: %.5f, D_recon: %.5f, D_orig: %.5f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss, d_recon_loss, d_orig_loss))

                # Balance the discriminator and generator
                if d_recon_loss < 0.2 and d_orig_loss < 0.2:
                    train_disc = False
                    train_vae = True
                elif d_recon_loss <0.5 and d_orig_loss <0.5:
                    train_disc = True
                    train_vae = True
                else:
                    train_disc = True
                    train_vae = False
                print('train_disc: %s, train_vae: %s' % (train_disc, train_vae))
                rec_loss = 0
                kl_loss = 0
                d_recon_loss = 0
                d_orig_loss = 0
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
        

        vae = vaegan.vae
        if (epoch+1)%50==0:
            for j, x in enumerate(test_loader):
                if type(x) is list:
                    x = x[0]
                vae.eval()

                out = x.data.cpu()
                # out = (out + 1) / 2
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/original%s.png' % (epoch), nrow=8)

                out = vae.decode(vae.encode(x)[0])  #out=x_tilde
                out = out.data.cpu()
                # out = (out + 1) / 2
                save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), directory+'/recon%s.png' % (epoch), nrow=8)

                # out = vae(None, 100)  ##out=x_p
                # out = out.data.cpu()
                # # out = (out + 1) / 2
                # save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './vae_result/generated%s.png' % (i), nrow=8)
                break
            
            mus_list = []
            for j, x in enumerate(test_loader):
                if type(x) is list:
                    x = x[0]
                mus = vae.encode(x)[0]
                mus_list.append(mus.data.cpu().numpy())
            mus_cat = np.concatenate(mus_list, axis=0)


            from scipy.stats import norm

            s = np.linspace(-3, 3, 300)

            fig = plt.figure(figsize=(20, 20))
            fig.subplots_adjust(hspace=0.6, wspace=0.4)

            for i in range(30):
                ax = fig.add_subplot(3, 10, i+1)
                ax.hist(mus_cat[:,i], density=True, bins = 20)
                ax.axis('off')
                ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
                ax.plot(s,norm.pdf(s))

            plt.savefig(directory+'/hist%s.png' % (epoch+1))

    print("Training finish!... save training results")
    torch.save(vae.state_dict(), model_name)

if __name__ == '__main__':
    main()
