import math
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from models.seq_mnist.qsnn_model import LightningSeqMNISTClassifier

parser = ArgumentParser()
parser.add_argument('--cpus-per-trial', type=int, default=3)
parser.add_argument('--gpus-per-trial', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--num-samples', type=int, default=10)
parser.add_argument('--data-dir', type=str, default='~/data')
parser.add_argument('--num-gpus', type=float, default=1.)
args = parser.parse_args()

kwargs = {
    'max_epochs': args.num_epochs,
    'precision': 16,
    "gpus": math.ceil(args.num_gpus),
    'logger': CSVLogger('logs'),
}

trainer = pl.Trainer(**kwargs)

config = {
    "base_weight": 0.015599835410916717,
    "batch_size": 256,
    "encouragement_strategy": "none",
    "l1_coeff": 0.0001,
    "l2_coeff": 0.0,
    "lr": 0.1,
    "max_delay": 10,
    "max_fan_in": 128,
    "min_nonzero_connections": 1,
    "momentum": 0.9,
    "n_adaptive_neurons": 120,
    "n_regular_neurons": 144,
    "refractory": 5,
    "removal_strategy": "l1_sparsity_and_random_removal",
    "target_connectivity": 0.2,
    "target_connectivity_epsilon": 0.0,
    "tau_trace": 20,
    "thresholds": 40,
    "weight_gain": 1.1542045268438628
}

model = LightningSeqMNISTClassifier(config=config, data_dir=args.data_dir)
trainer.fit(model)
trainer.test()
