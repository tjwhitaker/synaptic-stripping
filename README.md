# Synaptic Stripping

This repository is the official implementation of [Synaptic Stripping](#). This codebase is in progress of being cleaned up for eventual open source release.

![](./figures/neuroregeneration.svg)

Dead neurons are identified. Problematic connections are pruned. Training continues with regenerated neurons.

The current implementation is set up for training vision transformers and testing with corrupted datasets.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

```train
python train.py --checkpoint_dir=<STR> \
                --dataset=<STR> \
                --data_path=<STR> \
                --batch_size=<INT> \
                --num_workers=<INT> \
                --model=<STR> \
                --activation=<STR> \
                --patch_size=<INT> \
                --heads=<INT> \
                --layers=<INT> \
                --hidden_size=<INT> \
                --expansion_factor=<INT> \
                --epochs=<INT> \
                --warmup_epochs=<INT> \
                --save_freq=<INT> \
                --optimizer=<STR> \
                --scheduler=<STR> \
                --init_lr=<FLOAT> \
                --final_lr=<FLOAT> \
                --weight_decay=<FLOAT> \
                --autoaugment=<BOOL> \
                --synaptic_stripping=<BOOL> \
                --stripping_frequency=<INT> \
                --stripping_factor=<FLOAT> \
                --verbose=<BOOL> \
                --seed=<INT> \
```

Parameters:

- `checkpoint_dir` &mdash; where to save the trained model files (default: checkpoints/)
- `dataset` &mdash; name of the dataset to use [cifar10/cifar100/svhn/tinyimagenet] (default: cifar10)
- `data_path` &mdash; where to download the datasets (default: data/)
- `batch_size` &mdash; batch size for both the training and test loaders (default: 128)
- `num_workers` &mdash; number of workers to use for the data loaders (default: 4)
- `model` &mdash; name of the model architecture [vit/~~mlp~~] (default: vit)
- `activation` &mdash; name of activation function in transformer encoder MLPs [gelu/relu] (default: relu)
- `patch_size` &mdash; size of image patches (default: 8)
- `heads` &mdash; number of heads in the self attention layers (default: 8)
- `layers` &mdash; number of transformer encoder layers (default: 7)
- `hidden_size` &mdash; number of features in the self attention layers (default: 384)
- `expansion_factor` &mdash; expansion of the hidden size for mlp layers in the encoder blocks (default: 4)
- `epochs` &mdash; number of training epochs (default: 200)
- `warmup_epochs` &mdash; number of epochs to linearly increase the learning rate from 0 to init_lr (default: 5)
- `save_freq` &mdash; save the model every n epochs during training (default: -1)
- `optimizer` &mdash; optimizer to use for training the parent network [adam/~~sgd~~] (default: adam)
- `scheduler` &mdash; learning rate schedule for the main phase of training [cosine/~~linear~~] (default: cosine)
- `init_lr` &mdash; initial learning rate (default: 1e-3)
- `final_lr` &mdash; final learning rate (default: 1e-5)
- `weight_decay` &mdash; weight decay for the optimizer (default: 5e-5)
- `autoaugment` &mdash; include autoaugment data augmentation [true/false] (default: true)
- `synaptic_stripping` &mdash; include synaptic stripping during training [true/false] (default: true)
- `stripping_frequency` &mdash; number of epochs between synaptic stripping iterations (default: 1)
- `stripping_factor` &mdash; percentage of remaining weights to prune from dead neurons at each stripping iteration [0 < x < 1.0] (default: 0.05)
- `verbose` &mdash; [true/false] (default: true)
- `seed` &mdash; (default: 1)

## Evaluation

Evaluation on corrupted versions of CIFAR-10, CIFAR-100, SVHN, and Tiny ImageNet. Documentation in progress...

Expected directory setup:

```
data
  - CIFAR-10-C
  - CIFAR-100-C
  - SVHN-C
  - TINY-IMAGENET-C
```
