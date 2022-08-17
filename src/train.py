import argparse
import torch
import torchmetrics
import os
import sys
from models.vit import ViT
from data import get_loaders
from utils import setup_hooks, remove_hooks, calculate_dead_neurons, synaptic_strip, count_active_parameters

parser = argparse.ArgumentParser(description='Synaptic Stripping')

parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', metavar='STR',
                    help='training directory (default: checkpoints)')
parser.add_argument('--seed', type=int, default=1,
                    metavar='INT', help='random seed (default: 1)')
parser.add_argument('--verbose', type=int, default=1,
                    metavar='INT', help='logging verbosity (default: 1)')

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
parser.add_argument('--model', type=str, default="ViT", metavar='STR',
                    help='model name (default: vit)')
parser.add_argument('--activation', type=str, default='relu', metavar='STR',
                    help='activation function for transformer encoder MLPs (default: relu)')
parser.add_argument('--patch_size', type=int, default=8,
                    metavar='INT', help='image patch size (default: 8)')
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
parser.add_argument('--autoaugment', type=int, default=1, metavar='INT',
                    help='use autoaugment data augmentation (default: 1)')

# Stripping Args
parser.add_argument('--synaptic_stripping', type=int, metavar='INT',
                    default=1, help='use synaptic stripping (default: 1)')
parser.add_argument('--stripping_frequency', type=int, default=1, metavar='INT',
                    help='number of epochs in between stripping iterations (default: 1)')
parser.add_argument('--stripping_factor', type=float, default=0.05, metavar='FLOAT',
                    help='percentage of weights to remove from dead neurons at each stripping iteration (default: 0.05)')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.verbose:
    print(''.join(f"{k}={v}\n" for k, v in vars(args).items()))

#######
# Data
#######

os.makedirs(args.checkpoint_dir, exist_ok=True)

train_loader, test_loader = get_loaders(
    dataset=args.dataset,
    data_path=args.data_path,
    autoaugment=args.autoaugment,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

########
# Model
########

model = ViT(
    patch=args.patch_size,
    head=args.heads,
    num_layers=args.layers,
    hidden=args.hidden_size,
    mlp_hidden=args.hidden_size * args.expansion_factor,
    num_classes=len(train_loader.dataset.classes),
    activation=args.activation).to(device)

########
# Train
########

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-5, end_factor=1, total_iters=args.warmup_epochs)
decay = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.final_lr)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs])

# Metrics
train_accuracy = torchmetrics.Accuracy().to(device)
train_loss = torchmetrics.MeanMetric().to(device)
test_accuracy = torchmetrics.Accuracy().to(device)
test_loss = torchmetrics.MeanMetric().to(device)
num_dead_neurons = torchmetrics.SumMetric().to(device)
num_active_params = torchmetrics.SumMetric().to(device)

if args.synaptic_stripping:
    hook_outputs = setup_hooks(model)

for epoch in range(args.epochs):
    # Training Loop
    model.train()
    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)

        train_accuracy.update(predictions, targets)

        loss = criterion(predictions, targets)
        train_loss.update(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation Loop
    model.eval()

    if args.activation == 'relu':
        hook_handles, hook_outputs = setup_hooks(model)

    for _, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)

        test_accuracy.update(predictions, targets)

        loss = criterion(predictions, targets)
        test_loss.update(loss.item())

    if args.activation == 'relu':
        dead_neurons = calculate_dead_neurons(hook_outputs)
        remove_hooks(hook_handles)

        for layer, indices in dead_neurons.items():
            num_dead_neurons.update(len(indices))

        if args.synaptic_stripping and (epoch % args.stripping_frequency == 0):
            synaptic_strip(model, dead_neurons, args.stripping_factor, device)

        num_active_params.update(count_active_parameters(
            model, dead_neurons, args.synaptic_stripping, device))

    # Debug
    scheduler.step()

    if (epoch % args.save_freq == 0) or (epoch == (args.epochs - 1)):
        torch.save(model.state_dict(
        ), f"{args.checkpoint_dir}/{args.activation}_{args.synaptic_stripping}_{epoch}.pt")

    total_train_accuracy = train_accuracy.compute()
    total_train_loss = train_loss.compute()
    total_test_accuracy = test_accuracy.compute()
    total_test_loss = test_loss.compute()
    total_dead_neurons = num_dead_neurons.compute()
    total_active_params = num_active_params.compute()

    train_accuracy.reset()
    train_loss.reset()
    test_accuracy.reset()
    test_loss.reset()
    num_dead_neurons.reset()
    num_active_params.reset()

    if args.verbose:
        print("=============================================================")
        print()
        print(f"Epoch: {epoch}")
        print(
            f"Train Loss: {total_train_loss} Train Accuracy: {total_train_accuracy}")
        print(
            f"Test Loss: {total_test_loss} Test Accuracy: {total_test_accuracy}")
        print(f"Number Dead Neurons: {total_dead_neurons}")
        print(f"Number of Active Parameters: {total_active_params}")
        print()
