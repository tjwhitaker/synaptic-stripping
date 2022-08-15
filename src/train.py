import argparse
import torch
import os
import sys
from models.vit import ViT
from data import get_loaders
from utils import setup_hooks, zero_hooks

parser = argparse.ArgumentParser(description='Synaptic Stripping')

parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', metavar='STR',
                    help='training directory (default: checkpoints)')
parser.add_argument('--seed', type=int, default=1,
                    metavar='INT', help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True,
                    metavar='BOOL', help='logging verbosity (default: True)')

# Data Args
parser.add_argument('--dataset', type=str, default='cifar10', metavar='STR',
                    help='dataset name (default: cifar10)')
parser.add_argument('--data_path', type=str, default='data', metavar='STR',
                    help='path to datasets location (default: data)')
parser.add_argument('--batch_size', type=int, default=128, metavar='INT',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='INT',
                    help='number of workers (default: 4)')

# Model Args
parser.add_argument('--model', type=str, default=None, metavar='STR',
                    help='model name (default: vit)')
parser.add_argument('--activation', type=str, default='relu', metavar='STR',
                    help='activation function for transformer encoder MLPs (default: relu)')
parser.add_argument('--heads', type=int, default=8, metavar='INT',
                    help='number of self attention heads (default: 8')
parser.add_argument('--layers', type=int, default=7, metavar='INT',
                    help='number of transformer encoder layers (default: 7)')
parser.add_argument('--hidden_size', type=int, default=384, metavar='INT',
                    help='number of features in the self attention layers (default: 384)')
parser.add_argument('--expansion_factor', type=int, default=4, metavar='INT',
                    help='expansion of the hidden size for MLP layers in the encoder blocks (default: 4)')

# Training Args
parser.add_argument('--epochs', type=int, default=200, metavar='INT',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='INT',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--save_freq', type=int, default=10, metavar='INT',
                    help='save frequency (default: 10)')
parser.add_argument('--optimizer', type=str, default='adam', metavar='STR',
                    help='optimizer (default: adam)')
parser.add_argument('--scheduler', type=str, default='cosine', metavar='STR',
                    help='name of the learning rate scheduler (default: cosine)')
parser.add_argument('--init_lr', type=float, default=1e-3, metavar='FLOAT',
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--final_lr', type=float, default=1e-5, metavar='FLOAT',
                    help='final learning rate (default: 1e-5)')
parser.add_argument('--weight_decay', type=float, default=5e-5, metavar='FLOAT',
                    help='weight decay (default: 5e-5)')
parser.add_argument('--autoaugment', type=bool, default=True, metavar='BOOL',
                    help='use autoaugment data augmentation (default: True)')

# Stripping Args
parser.add_argument('--synaptic_stripping', type=bool, metavar='BOOL',
                    default=True, help='use synaptic stripping (default: True)')
parser.add_argument('--stripping_frequency', type=int, default=1, metavar='INT',
                    help='number of epochs in between stripping iterations (default: 1)')
parser.add_argument('--stripping_factor', type=float, default=0.05, metavar='FLOAT',
                    help='percentage of weights to remove from dead neurons at each stripping iteration (default: 0.05)')

args = parser.parse_args()

torch.manual_seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######
# Data
#######

os.makedirs(args.checkpoint_dir, exist_ok=True)

(train_loader, test_loader), num_classes = get_loaders(
    dataset=args.dataset,
    data_path=args.data_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

model = ViT().to(device)

########
# Train
########

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8, end_factor=1, total_iters=args.warmup_epochs)
decay = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.final_lr)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs])

if args.synaptic_stripping:
    hook_outputs = setup_hooks(model)

for epoch in range(args.epochs):
    train_loss = 0
    test_loss = 0

    # Training Loop
    model.train()
    for _, (inputs, targets) in enumerate(train_loader):
        if args.synaptic_stripping:
            # Hooks automatically keep track of all forward pass outputs.
            # We zero out the hooks to prevent overflow.
            zero_hooks(hook_outputs)
        pass

    # Validation Loop
    model.eval()
    for _, (inputs, targets) in enumerate(test_loader):
        pass
    pass
