import torch
from torch import nn
from torch.nn import functional as F


class CVAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=1,filter_base=128,batch_norm=True):
        super(CVAE, self).__init__()

        d = filter_base
        self.d = d
        self.zsize = zsize
        self.batch_norm = batch_norm
        self.layer_count = layer_count

        mul = 1
        inputs = 2
        for i in range(self.layer_count):
            setattr(self, "conv_xy_%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv_xy_%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2
        # mul 1 -> 2 -> 4 -> 8 -> 16 (final channel size) -> 32
        mul = 1
        inputs = 1
        for i in range(self.layer_count):
            setattr(self, "conv_y_%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv_y_%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2
        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        self.fc3 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc4 = nn.Linear(inputs * 4 * 4, zsize)

        self.d1 = nn.Linear(zsize, inputs * 4 * 4)
        mul = inputs // d // 2
        inputs = inputs+1 # add one for the y channel
        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul +1))
            inputs = d * mul + 1
            mul //= 2
        # mul  16 -> 8 -> 4 -> 2 -> 1
        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))
        

    def encode_xy(self, x, y):
        # concat x and y into (batch,2,128,128)
        x = torch.cat((x,y),1)
        for i in range(self.layer_count):
            if self.batch_norm:
                x = F.relu(getattr(self, "conv_xy_%d_bn" % (i + 1))(getattr(self, "conv_xy_%d" % (i + 1))(x)))
            else:
                x = F.relu((getattr(self, "conv_xy_%d" % (i + 1))(x)))

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x) # mu0
        h2 = self.fc2(x) # logvar0
        return h1, h2

    def encode_y(self, y):
        for i in range(self.layer_count):
            if self.batch_norm:
                y = F.relu(getattr(self, "conv_y_%d_bn" % (i + 1))(getattr(self, "conv_y_%d" % (i + 1))(y)))
            else:
                y = F.relu((getattr(self, "conv_y_%d" % (i + 1))(y)))
        y = y.view(y.shape[0], self.d_max * 4 * 4)
        h3 = self.fc3(y) # mu1
        h4 = self.fc4(y) # logvar1
        return h3, h4
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z,y):
        z = z.view(z.shape[0], self.zsize)
        z = self.d1(z)
        z = z.view(z.shape[0], self.d_max, 4, 4)
        #z = self.deconv1_bn(x)
        y64 = torch.max_pool2d(y,kernel_size=2,stride=2)
        y32 = torch.max_pool2d(y64,kernel_size=2,stride=2)
        y16 = torch.max_pool2d(y32,kernel_size=2,stride=2)
        y8 = torch.max_pool2d(y16,kernel_size=2,stride=2)
        y4 = torch.max_pool2d(y8,kernel_size=2,stride=2)
        y_list = [y4,y8,y16,y32,y64,y]
        x = F.leaky_relu(z, 0.2)
        x = torch.cat((x,y4),1)
        for i in range(1, self.layer_count): # i = 1,2,3,4
            x = getattr(self, "deconv%d" % (i + 1))(x)
            x =  torch.cat((x,y_list[i]),1)
            if self.batch_norm:
                x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(x),0.2)
            else:
                x = F.leaky_relu(x,0.2)
        x = F.sigmoid(getattr(self, "deconv%d" % (self.layer_count + 1))(x)+y)
        return x

    def forward(self, x,y):
        mu0, logvar0 = self.encode_xy(x,y)
        mu1, logvar1 = self.encode_y(y) 
        mu0 = mu0.squeeze()
        logvar0 = logvar0.squeeze()
        mu1 = mu1.squeeze()
        logvar1 = logvar1.squeeze()
        z = self.reparameterize(mu0, logvar0)
        x_tilde = self.decode(z.view(-1, self.zsize, 1, 1),y)
        return x_tilde, mu0, logvar0, mu1, logvar1

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()