# ECG_GAN_MBD
This repository is for the paper "Synthesis of Realistic ECG using Generative Adversarial Networks". 

The current files uploaded are for implementing Minibatch Discrimination (MBD) for a 2 Layer CNN discriminator, please note that for ECG data with MBD layers the training does not converge.

You can edit the ```Model.py``` file accordingly to remove MBD layers and/or to add more Convolution-Pooling layer as described in the paper.

Usage:
```$python3 train.py```

------
## Citation

If you find this repo helpful in any way please cite our arXiv preprint:


    @misc{delaney2019synthesis,
      title={Synthesis of Realistic ECG using Generative Adversarial Networks},  
      author={Anne Marie Delaney and Eoin Brophy and Tomas E. Ward},
      year={2019},
      eprint={1909.09150},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
    }
