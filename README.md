# Quantized rewiring: hardware-aware training of sparse deep neural networks

This is the official code for the
paper ["Quantized rewiring: hardware-aware training of sparse deep neural networks"](https://doi.org/10.1088/2634-4386/accd8f)
for training sparse deep neural networks while considering hardware limitations.

"Quantized rewiring: hardware-aware training of sparse deep neural networks."  
*Petschenig, Horst and Robert Legenstein.*  
Neuromorphic Computing and Engineering 3.2 (2023): 024006.  
[https://doi.org/10.1088/2634-4386/accd8f](https://doi.org/10.1088/2634-4386/accd8f)

## Setup

You will have to install PyTorch and PyTorch Lightning to run this code. The dependencies are listed
in [environment.yml](environment.yml). If you use Conda, you can install the environment via

    conda env create -f environment.yml --name quantized_rewiring

to install all required packages and dependencies.

## Sequential MNIST

In this task we have tested the applicability of our approach on the well-known sequential MNIST benchmark which is a
difficult temporal credit-assignment problem. To start training, run 

    python train_seq_mnist.py

The `logs` directory will contain `.csv` files that track the training progress.