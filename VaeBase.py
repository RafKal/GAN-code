import torch
import torch.nn as nn

from models.utils.pt_utils import (
    kl_divergence_loss,
    TimeDistributed,
    repeat_vector,
    create_clipnorm_callabck,
    BestModelSaver,
    EarlyStopper,
)
import models.utils.pt_training_loops as training_loops
from config.config_bases import ModelConfigBase
from torch.utils.data import DataLoader
from tqdm import tqdm


class VaeBaseConfig(ModelConfigBase):
    def __init__(self):
        self.sequence_length = 50
        self.latent_dim = 256
        self.batch_size = 32
        self.learn_rate = 0.0001
        self.alpha = 10
        self.beta = 0.1
        self.n_channels = 0
        self.clipnorm = None


class ELBO(nn.Module):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def forward(self, x, y_pred, mu, log_var):
        x_reshape = x.contiguous().view(-1, x.size(-1) * x.size(-2))
        y_pred_reshape = y_pred.contiguous().view(-1, x.size(-1) * x.size(-2))
        squared_error = torch.square(y_pred_reshape - x_reshape)
        summed_error = torch.sum(squared_error, dim=1)
        mean_summed_squared_error = torch.mean(summed_error)
        kl_loss = kl_divergence_loss(mu, log_var)
        return (
            self.alpha * mean_summed_squared_error + self.beta * kl_loss,
            mean_summed_squared_error,
            kl_loss,
        )


class VaeBase(nn.Module):
    def __init__(self, model_config: VaeBaseConfig, device):
        super().__init__()
        self.device = device
        self.model_config = model_config
        self.tracker = ElboTracker()
        self.clipnorm = create_clipnorm_callabck(self.model_config.clipnorm)

        self._init_network()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.model_config.learn_rate
        )

    def _init_network(self):
        # ----------VAE-Params------------#
        n_channels = self.model_config.n_channels
        num_layers = 4  # lstm layers
        lstm_dim = 256  # equals keras units
        z_dim = self.model_config.latent_dim
        self.loss_fn = ELBO(self.model_config.alpha, self.model_config.beta)

        # ----------Encoder---------------#
        self.lstm_encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mu = nn.Linear(lstm_dim, z_dim)
        self.logvar = nn.Linear(lstm_dim, z_dim)

        # ----------Decoder---------------#
        self.repeat_factor = self.model_config.sequence_length
        self.lstm_decoder = nn.LSTM(
            input_size=z_dim,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.tdd_linear = TimeDistributed(
            nn.Linear(in_features=lstm_dim, out_features=n_channels)
        )

    def encoder(self, x):
        output, _ = self.lstm_encoder(x)
        h_n = output[:, -1, :]
        mu_raw, log_var_raw = torch.squeeze(self.mu(h_n)), torch.squeeze(
            self.logvar(h_n)
        )
        return mu_raw, log_var_raw

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def decoder(self, z):
        z_r = repeat_vector(z, self.model_config.sequence_length)
        output, _ = self.lstm_decoder(z_r)
        reconstructed = self.tdd_linear(output)
        return reconstructed

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        y_pred = self.decoder(z)
        return y_pred, mu, log_var

    def test_step(self, x):
        with torch.no_grad():
            y_pred, mu, log_var = self.forward(x)
            total_loss, mse, kl_loss = self.loss_fn(x, y_pred, mu, log_var)
            self.tracker.add_to_test_loss(total_loss, mse, kl_loss)

    def train_step(self, x):
        self.optimizer.zero_grad()
        y_pred, mu, logvar = self.forward(x)
        total_loss, mse, kl_loss = self.loss_fn(x, y_pred, mu, logvar)
        total_loss.backward()
        self.clipnorm(self.parameters(), self.model_config.clipnorm)
        self.optimizer.step()
        self.tracker.add_to_train_loss(total_loss, mse, kl_loss)

    def synthesize_data(self):
        z = torch.randn(self.model_config.batch_size, self.model_config.latent_dim).to(
            self.device
        )
        z_d = self.decoder(z)
        return z_d.cpu().detach()

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int,
        gpu_device: str,
        callbacks,
    ):
        return training_loops.fit_with_early_stopping(
            self, train_loader, test_loader, n_epochs, gpu_device, callbacks
        )


class ElboTracker:
    def __init__(self):
        self.init_trackers()

    def get_history(self):
        return {
            "train_loss": self.train_loss_history,
            "train_mse": self.train_mse_loss_history,
            "train_kl": self.train_kl_loss_history,
            "test_loss": self.test_loss_history,
            "test_mse": self.test_mse_loss_history,
            "test_kl": self.test_kl_loss_history,
        }

    def init_trackers(self):
        self.train_loss_history = list()
        self.test_loss_history = list()
        self.train_mse_loss_history = list()
        self.test_mse_loss_history = list()
        self.train_kl_loss_history = list()
        self.test_kl_loss_history = list()
        self.reset_epoch_trackers()

    def update_train_loss(self, n_batches):
        self.train_loss_history.append(self.sum_train_loss / n_batches)
        self.train_mse_loss_history.append(self.sum_mse_train_loss / n_batches)
        self.train_kl_loss_history.append(self.sum_kl_train_loss / n_batches)

    def update_test_loss(self, n_batches):
        self.test_loss_history.append(self.sum_test_loss / n_batches)
        self.test_mse_loss_history.append(self.sum_mse_test_loss / n_batches)
        self.test_kl_loss_history.append(self.sum_kl_test_loss / n_batches)

    def add_to_train_loss(self, total_loss, recon_loss, kl_loss):
        self.sum_train_loss += total_loss.item()
        self.sum_mse_train_loss += recon_loss.item()
        self.sum_kl_train_loss += kl_loss.item()

    def reset_epoch_trackers(self):
        self.sum_train_loss = 0
        self.sum_test_loss = 0
        self.sum_mse_train_loss = 0
        self.sum_mse_test_loss = 0
        self.sum_kl_train_loss = 0
        self.sum_kl_test_loss = 0

    def add_to_test_loss(self, total_loss, recon_loss, kl_loss):
        self.sum_test_loss += total_loss.item()
        self.sum_mse_test_loss += recon_loss.item()
        self.sum_kl_test_loss += kl_loss.item()


class VaeBaseBottleneckedConfig(VaeBaseConfig):
    def __init__(self):
        return


class VaeBaseBottlenecked(VaeBase):
    def __init__(self, model_config: VaeBaseBottleneckedConfig, device):
        super().__init__(model_config, device)

    def _init_network(self):
        n_c = self.model_config.n_channels
        z_dim = self.model_config.latent_dim
        self.loss_fn = ELBO(self.model_config.alpha, self.model_config.beta)

        # ----------Encoder---------------#
        self.lstm_encoder = nn.Sequential(
            nn.LSTM(input_size=n_c, hidden_size=256, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True),
        )

        self.mu = nn.Linear(32, z_dim)
        self.logvar = nn.Linear(32, z_dim)

        # ----------Decoder---------------#
        self.repeat_factor = self.model_config.sequence_length
        self.lstm_decoder = nn.Sequential(
            nn.LSTM(input_size=z_dim, hidden_size=32, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True),
            GetLSTMOutput(),
            nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True),
        )
        self.tdd_linear = TimeDistributed(nn.Linear(in_features=256, out_features=n_c))


class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out
