# ECG_GAN_MBD
This repository is for the paper "Synthesis of Realistic ECG using Generative Adversarial Networks". 

The current files uploaded are for implementing Minibatch Discrimination (MBD) for a 2 Layer CNN discriminator, please note that for ECG data with MBD layers the training does not converge.

You can edit the Model.py file accordingly to remove MBD layers and/or to add more Convolution-Pooling layer as described in the paper.

Run the script by:
$python3 train.py
