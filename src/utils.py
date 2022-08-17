import torch
import torch.nn.utils.prune
from functools import reduce


def forward_hook(name, hook_outputs):
    def hook(module, input, output):
        sum_over_batches = torch.sum(output.detach(), dim=0)
        if name in hook_outputs:
            hook_outputs[name] = torch.add(
                hook_outputs[name], sum_over_batches)
        else:
            hook_outputs[name] = sum_over_batches
    return hook


def setup_hooks(model):
    hook_outputs = {}
    hook_handles = []

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU):
            handle = layer.register_forward_hook(
                forward_hook(name, hook_outputs))
            hook_handles.append(handle)

    return hook_handles, hook_outputs


def remove_hooks(handles):
    for h in handles:
        h.remove()


def calculate_dead_neurons(hook_outputs):
    dead_neurons = {}

    for (name, activations) in hook_outputs.items():
        dead_neuron_indices = []
        sum_over_embedding = torch.sum(activations, dim=0)

        for i in range(len(sum_over_embedding)):
            if sum_over_embedding[i] == 0:
                dead_neuron_indices.append(i)

        dead_neurons[name] = dead_neuron_indices

    return dead_neurons


def synaptic_strip(model, dead_neurons, stripping_factor, device):
    for (name, indices) in dead_neurons.items():
        module_layers = name.split('.')

        # Hacky way to get the linear layer since hooks use the relu.
        # Substract 1 to index the linear layer preceding the relu in sequential module.
        module_layers[-1] = str(int(module_layers[-1]) - 1)
        layer = reduce(getattr, module_layers, model)

        if isinstance(layer, torch.nn.Linear):
            mask = torch.ones(layer.weight.shape).to(device)

            for dead_neuron in indices:
                values = layer.weight[dead_neuron]
                nonzeros = values[values != 0]
                threshold = torch.quantile(nonzeros, q=stripping_factor)

                mask[dead_neuron][layer.weight[dead_neuron] < threshold] = 0

            torch.nn.utils.prune.custom_from_mask(layer, 'weight', mask)

            # Hacky memory leak fix from repeated pruning
            for k in list(layer._forward_pre_hooks):
                hook = layer._forward_pre_hooks[k]
                if isinstance(hook, torch.nn.utils.prune.PruningContainer):
                    if isinstance(hook[-1], torch.nn.utils.prune.CustomFromMask):
                        hook[-1].mask = None
                elif isinstance(hook, torch.nn.utils.prune.CustomFromMask):
                    hook.mask = None


def count_active_parameters(model, dead_neurons, synaptic_stripping, device):
    active_params = 0

    for (name, indices) in dead_neurons.items():
        module_layers = name.split('.')

        # Hacky way to get the linear layer since hooks use the relu.
        # Substract 1 to index the linear layer preceding the relu in sequential module.
        module_layers[-1] = str(int(module_layers[-1]) - 1)
        layer = reduce(getattr, module_layers, model)

        if isinstance(layer, torch.nn.Linear):
            # Mask dead neurons
            mask = torch.ones(layer.weight.shape).to(device)
            mask[indices, :] = 0

            if synaptic_stripping:
                mask = torch.mul(mask, layer.weight_mask)

            active_params += torch.count_nonzero(mask)

    return active_params
