import math
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from layers.neuron_models import AdaIafPscDeltaModuleDelay
from layers.qsnn_models import QRecurrentSNNDelay
from models.seq_mnist import seq_mnist_dataset
from utils.transforms import ThresholdEncode, AddPrompt


class LightningSeqMNISTClassifier(pl.LightningModule):

    def __init__(self, config: dict, data_dir=None, cpus_per_trial=3, gpus_per_trial=1):
        super(LightningSeqMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()

        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.momentum = config['momentum']
        self.thresholds = config['thresholds']
        if 'n_neurons' in config:
            n_neurons = config['n_neurons']
            self.n_adaptive_neurons = int(n_neurons * 5. / 11.)
            self.n_regular_neurons = int(n_neurons * 6 / 11.)
        else:
            self.n_adaptive_neurons = config.get('n_adaptive_neurons', 100)
            self.n_regular_neurons = config.get('n_regular_neurons', 120)
        self.tau_trace = config['tau_trace']
        self.base_weight = config['base_weight']
        self.max_fan_in = config['max_fan_in']
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.target_firing_rate = config.get('target_firing_rate', 20e-3)
        self.firing_rate_coeff = config.get('firing_rate_coeff', 1e-1)
        self.voltage_coeff = config.get('voltage_coeff', 1e-3)
        self.l1_coeff = config.get('l1_coeff', 1e-4)
        self.l2_coeff = config.get('l2_coeff', 1e-4)
        self.quantized_readout = config.get('quantized_readout', False)

        self.rewiring_config = {
            'removal_strategy': config.get('removal_strategy', 'none'),
            'encouragement_strategy': config.get('encouragement_strategy', 'none'),
            'random_walk_strategy': config.get('random_walk_strategy', 'none'),
            'num_baseweights': 2,
            'base_weights': {
                'excitatory': self.base_weight,
                'inhibitory': -self.base_weight,
            },
            'max_fan_in': self.max_fan_in,
        }
        if 'target_connectivity' in config:
            self.rewiring_config['target_connectivity'] = config['target_connectivity']
        if 'target_connectivity' in config:
            self.rewiring_config['target_connectivity_epsilon'] = config['target_connectivity_epsilon']
        if 'weight_gain' in config:
            self.rewiring_config['weight_gain'] = config['weight_gain']
        if 'max_delay' in config and config['max_delay'] is not None:
            self.rewiring_config['max_delay'] = config['max_delay']
        if 'min_nonzero_connections' in config:
            self.rewiring_config['min_nonzero_connections'] = config['min_nonzero_connections']
        average_output_ms = config.get('average_output_ms', 56)

        self.criterion = nn.CrossEntropyLoss()

        input_size = 2 * self.thresholds + 1
        output_size = 10
        dynamics = AdaIafPscDeltaModuleDelay(n_adaptive_neurons=self.n_adaptive_neurons,
                                                 n_regular_neurons=self.n_regular_neurons, tau_adaptation=700.,
                                                 dampening_factor=0.3,
                                                 adaptation_constant=1.8, thr=0.01, refractory_timesteps=5)
        self.snn = QRecurrentSNNDelay(input_size, self.n_adaptive_neurons, self.n_regular_neurons, output_size,
                                      autapses=False,
                                      tau_trace=self.tau_trace,
                                      dynamics=dynamics, rewiring_config=self.rewiring_config,
                                      average_output_ms=average_output_ms, multi_baseweight=True)

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            ThresholdEncode(0., 1., self.thresholds),
            AddPrompt(2 * 28)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            ThresholdEncode(0., 1., self.thresholds),
            AddPrompt(2 * 28)
        ])

    def forward(self, x):
        out, spikes, trace, state = self.snn(x)
        return F.log_softmax(out), spikes, trace, state

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits, spikes, trace, state = self.forward(x)
        z_t, v_t, r_t, a_t = (state[0], state[1], state[2], state[3])
        loss_cross_entropy = self.cross_entropy_loss(logits, y)
        mean_firing_rate = torch.mean(spikes, dim=(0, 1))
        loss_firing_rate = self.firing_rate_coeff * torch.square(
            torch.sum(torch.square(mean_firing_rate - self.target_firing_rate)))
        loss_voltage_regularization = self.voltage_coeff * (0.5 * torch.sum(
            torch.square(torch.clamp(v_t - a_t, 0.)) + torch.square(torch.clamp(-v_t - a_t, 0.))))
        loss_l1 = self.l1_coeff * self.snn.input_recurrent_layer.weight.abs().sum()
        loss_l2 = self.l2_coeff * self.snn.input_recurrent_layer.weight.square().sum()
        total_loss = loss_cross_entropy + loss_firing_rate + loss_voltage_regularization + loss_l1 + loss_l2
        accuracy = self.accuracy(logits, y)
        fan_in_stats = self.snn.input_recurrent_layer.compute_fan_in_stats()
        connectivity_ = fan_in_stats['connectivity']
        assert connectivity_ <= self.rewiring_config['target_connectivity'] + 0.01

        self.log("ptl/train_loss", total_loss)
        self.log('ptl/train_loss_firing_rate', loss_firing_rate)
        self.log('ptl/train_loss_voltage_reg', loss_voltage_regularization)
        self.log("ptl/train_accuracy", accuracy)
        self.log('ptl/train_loss_l1', loss_l1)
        self.log('ptl/train_loss_l2', loss_l2)
        self.log('ptl/train_mean_firing_rate', torch.mean(mean_firing_rate) * 1e3)
        for k, v in fan_in_stats.items():
            self.log(f'ptl/fan_in_stats/{k}', v)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits, spikes, trace, _ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits, spikes, trace, _ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        self.log("ptl/test_loss", avg_loss)
        self.log("ptl/test_accuracy", avg_acc)


    def prepare_data(self):
        seq_mnist_dataset.prepare()

    def setup(self, stage=None):
        self.seq_mnist_train = seq_mnist_dataset.get_train(self.train_transform)
        self.seq_mnist_test = seq_mnist_dataset.get_test(self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.seq_mnist_train, batch_size=int(self.batch_size), num_workers=self.cpus_per_trial)

    def val_dataloader(self):
        return DataLoader(self.seq_mnist_test, batch_size=int(self.batch_size), num_workers=self.cpus_per_trial)

    def test_dataloader(self):
        return DataLoader(self.seq_mnist_test, batch_size=int(self.batch_size), num_workers=self.cpus_per_trial)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5),
            'name': 'reduce_lr_on_plateau_logger',
            'monitor': 'ptl/val_loss'
        }
        return [optimizer], [scheduler]
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # return [optimizer], [scheduler]


def train_seq_mnist_tune_checkpoint(config, checkpoint_dir=None, num_epochs=200, num_gpus=0, data_dir="~/data",
                                    precision=16):
    data_dir = os.path.expanduser(data_dir)
    kwargs = {
        "max_epochs": num_epochs,
        'precision': precision,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        # "logger": TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        "progress_bar_refresh_rate": 0,
        "callbacks": [
            LearningRateMonitor()
        ]
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")

    model = LightningSeqMNISTClassifier(config=config, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model)
