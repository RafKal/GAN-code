""" 
2nd stage of the GAN model, with a 2G 4D model 
"""

import torch.nn as nn
from config.config_bases import ModelConfigBase
import torch.optim as optim
import torch
import models.utils.pt_training_loops as training_loops
from torch.utils.data import DataLoader
from models.utils.pt_utils import repeat_vector
from models.utils.pt_utils import TimeDistributed


class GanConfig(ModelConfigBase):
    def __init__(self):
        self.lr = 3e-4
        self.n_channels = 0  # auto determined depending on dataset
        self.z_dim = 256  # dim of latent noise of generator, from 256
        self.batch_size = 128
        self.num_epochs = 20 
        self.hidden_layers = 256 
        


class Gan(nn.Module):
    def __init__(self, model_config: GanConfig, device):
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.tracker = GanLossTracker()
        
        
        self.disc = DiscriminatorLSTM(model_config.n_channels, model_config.batch_size, model_config.hidden_layers).to(device)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=model_config.lr)


        self.gen = Generator(model_config.z_dim, model_config.n_channels, model_config.hidden_layers).to(device)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=model_config.lr/3)
        
        self.criterion = nn.BCELoss()
        
     


        
    def forward(self):
        noise = torch.randn(self.model_config.batch_size, self.model_config.z_dim).to(
            self.device
        )
        noise = repeat_vector(noise, self.model_config.sequence_length)

        fake = self.gen(noise)
        return fake

    def test_step(self, real):
        return

    def train_step(self, real):
        fake = self.forward()


        #print(real.shape)
        #print(fake.shape)

        disc_real = self.disc(real)  # .view(-1)
        lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = self.disc(fake)  # .view(-1)
        lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        self.disc.zero_grad()
        lossD.backward(retain_graph=True)
        self.opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = self.disc(fake).view(-1)
        lossG = self.criterion(output, torch.ones_like(output))
        self.gen.zero_grad()
        lossG.backward()
        self.opt_gen.step()
        self.tracker.add_to_train_loss(lossD, lossG)
    def synthesize_data(self):
        return self.forward().cpu().detach()

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int,
        gpu_device: str,
        callbacks,
    ):
        return training_loops.fit(self, train_loader, test_loader, n_epochs, gpu_device)


class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        #print(f"out shape: {out.shape}")
        return out


class getLastTimestep(nn.Module):
    def forward(self, x):
        out = x[:, -1, :]
        return out

class Generator(nn.Module):
    def __init__(self, z_dim, data_dim, hidden_layers):
        super().__init__()
        self.gen = nn.Sequential(
            nn.LSTM(z_dim, hidden_layers, 1, batch_first=True),
            GetLSTMOutput(),       
            nn.LSTM(hidden_layers, hidden_layers,1, batch_first=True),
            GetLSTMOutput(),
           
            TimeDistributed(nn.Linear(in_features=hidden_layers, out_features=data_dim),
        )
        )
        

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, in_features, batch_size):
        super().__init__()
        self.disc = nn.Sequential(
            Reshape([batch_size, 50*in_features]), #flatten each batch
            nn.Linear(50*in_features, 3),
            nn.Linear(3, 1),
            nn.Sigmoid(),       
        )
        

    def forward(self, x):
        return self.disc(x)


class DiscriminatorLSTM(nn.Module):
    def __init__(self, data_dim, batch_size, hidden_layers):
        super().__init__()
        self.disc = nn.Sequential(
            nn.LSTM(data_dim, hidden_layers, batch_first=True),
            GetLSTMOutput(),
           
            nn.LSTM(hidden_layers, hidden_layers, 3, batch_first=True),
            GetLSTMOutput(),

            Discriminator(hidden_layers, batch_size),
        )

    def forward(self, x):
        return self.disc(x)


class GanLossTracker:
    def __init__(self):
        self.init_trackers()

    def get_history(self):
        return {
            "sum_loss_d": self.loss_d_history,
            "sum_loss_g": self.loss_g_history,
        }

    def init_trackers(self):
        self.loss_d_history = list()
        self.loss_g_history = list()
        self.reset_epoch_trackers()

    def update_train_loss(self, n_batches):
        self.loss_d_history.append(self.sum_loss_d / n_batches)
        self.loss_g_history.append(self.sum_loss_g / n_batches)

    def add_to_train_loss(self, loss_d, loss_g):
        self.sum_loss_d += loss_d.item()
        self.sum_loss_g += loss_g.item()

    def reset_epoch_trackers(self):
        self.sum_loss_d = 0
        self.sum_loss_g = 0

    def update_test_loss(self, n_batches):
        return



# Just in case its needed for the Discriminator
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        

    def forward(self, x):
        #print(x.shape)
        #print(x.view(self.shape).shape)
        #return x.view(self.shape)
        return torch.reshape(x, shape=self.shape)
    


class RepeatVector(nn.Module):
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def forward(self, x):
        #print(x.repeat(1, 1, self.n).shape)
        # t_a = torch.unsqueeze(x, 1)
        # t_b = t_a.repeat(1, self.n, 1)
        # return t_b
    
        return x.repeat(1, self.n, 1)
        
class print_output(nn.Module):
    def __init__(self, text, inc_value):
        super(print_output, self).__init__()
        self.text = text
        self.inc_value = inc_value

    def forward(self, x): 
        print(f'{self.text}: {x.shape}')
        if self.inc_value:
            print(x)

        return x
    

class Swap(nn.Module):
    def __init__(self, *dims):
        super(Swap, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)



