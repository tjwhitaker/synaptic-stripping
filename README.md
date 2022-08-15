# Synaptic Stripping

This repository is the official implementation of [Synaptic Stripping](#). This codebase is still in progress of being cleaned up for eventual open source release.

![](./figures/neuroregeneration.svg)

The current implementation is set up for vision transformers.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

```train
python train.py --checkpoint_dir=<DIR> \
                --dataset=<DATASET> \
                --data_path=<DATA_PATH> \
                --batch_size=<BATCH_SIZE> \
                --num_workers=<NUM_WORKERS> \
                --model=<MODEL> \
                --heads=<HEADS> \
                --layers=<LAYERS> \
                --hidden_size=<HIDDEN_SIZE> \
                --expansion_factor=<EXPANSION_FACTOR> \
                --epochs=<EPOCHS> \
                --warmup_epochs=<WARMUP_EPOCHS> \
                --save_freq=<SAVE_FREQ> \
                --optimizer=<OPTIMIZER> \
                --scheduler=<SCHEDULER> \
                --init_lr=<INIT_LR> \
                --final_lr=<FINAL_LR> \
                --weight_decay=<WEIGHT_DECAY> \
                --autoaugment=<AUTOAUGMENT> \
                --stripping_frequency=<STRIPPING_FREQUENCY> \
                --stripping_factor=<STRIPPING_FACTOR> \
                --verbose=<V> \
                --seed=<SEED> \
```

Parameters:

- `checkpoint_dir` &mdash; where to save the trained model files (default: checkpoints/)
- `dataset` &mdash; name of the dataset to use [cifar10/cifar100/svhn/tinyimagenet] (default: cifar10)
- `data_path` &mdash; where to download the datasets (default: data/)
- `batch_size` &mdash; batch size for both the training and test loaders (default: 128)
- `num_workers` &mdash; number of workers to use for the data loaders (default: 4)
- `model` &mdash; name of the model architecture [vit/~~mlp~~] (default: vit)
- `heads` &mdash; number of heads in the self attention layers (default: 8)
- `layers` &mdash; number of transformer encoder layers (default: 7)
- `hidden_size` &mdash; number of features in the self attention layers (default: 384)
- `expansion_factor` &mdash; expansion of the hidden size fo mlp layers in the encoder blocks (default: 4)
- `epochs` &mdash; number of training epochs (default: 200)
- `warmup_epochs` &mdash; number of epochs to linearly increase the learning rate from 0 to init_lr (default: 5)
- `save_freq` &mdash; save the model every n epochs during training (default: 10)
- `optimizer` &mdash; optimizer to use for training the parent network [adam/~~sgd~~] (default: adam)
- `scheduler` &mdash; learning rate schedule for the main phase of training [cosine/~~linear~~] (default: cosine)
- `init_lr` &mdash; initial learning rate (default: 1e-3)
- `final_lr` &mdash; final learning rate (default: 1e-5)
- `weight_decay` &mdash; (default: 5e-5)
- `autoaugment` &mdash; include autoaugment data augmentation [true/false] (default: true)
- `stripping_frequency` &mdash; number of epochs between synaptic stripping iterations (default: 1)
- `stripping_factor` &mdash; percentage of remaining weights to prune from dead neurons at each stripping iteration [0 < x < 1.0] (default: 0.05)
- `verbose` &mdash; [true/false] (default: true)
- `seed` &mdash; (default: 1)
