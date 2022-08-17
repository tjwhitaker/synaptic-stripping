import argparse
import torch
import torch.nn.utils.prune as prune
from models.vit import ViT
from data import get_corrupted_loader
from functools import reduce
import torchmetrics

parser = argparse.ArgumentParser(description='Synaptic Stripping')

parser.add_argument('--checkpoint', type=str, default=None, metavar='STR',
                    help='path to model file (default: None)')
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

# Stripping Args
parser.add_argument('--synaptic_stripping', type=int, metavar='INT',
                    default=1, help='use synaptic stripping (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.verbose:
    print(''.join(f"{k}={v}\n" for k, v in vars(args).items()))

#######
# Data
#######

loader = get_corrupted_loader(
    dataset=args.dataset,
    data_path=args.data_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers)

num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'tinyimagenet': 200
}

########
# Model
########

model = ViT(
    patch=args.patch_size,
    head=args.heads,
    num_layers=args.layers,
    hidden=args.hidden_size,
    mlp_hidden=args.hidden_size * args.expansion_factor,
    num_classes=num_classes[args.dataset],
    activation=args.activation).to(device)

criterion = torch.nn.CrossEntropyLoss()

# Empty prune in order to load state dict for stripped models
if args.synaptic_stripping:
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU):
            module_layers = name.split('.')

            # Hacky way to get the linear layer since hooks use the relu.
            # Substract 1 to index the linear layer preceding the relu in sequential module.
            module_layers[-1] = str(int(module_layers[-1]) - 1)
            layer = reduce(getattr, module_layers, model)

            if isinstance(layer, torch.nn.Linear):
                prune.random_unstructured(layer, "weight", amount=0.0)


model.load_state_dict(torch.load(args.checkpoint))

##########
# Metrics
##########

accuracy = torchmetrics.Accuracy().to(device)
loss = torchmetrics.MeanMetric().to(device)

model.eval()
with torch.no_grad():
    num_correct = 0
    for _, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)

        accuracy.update(predictions, targets)
        loss.update(criterion(predictions, targets).item())

if args.verbose:
    print("=============================================================")
    print()
    print(f"Loss: {loss.compute()} Accuracy: {accuracy.compute()}")
    print()
