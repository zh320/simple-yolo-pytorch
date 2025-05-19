import torch.nn as nn


def prune_weights(layer_weights, channel_sparsity, is_first_layer=False):
    out_channels = layer_weights.shape[0]
    in_channels = layer_weights.shape[1]

    new_out_channels = int(out_channels * channel_sparsity)

    if is_first_layer:
        # For the first layer, only prune the output channels, not the input channels
        pruned_weights = layer_weights[:new_out_channels]
    else:
        # For all other layers, prune both input and output channels
        new_in_channels = int(in_channels * channel_sparsity)
        pruned_weights = layer_weights[:new_out_channels, :new_in_channels]

    return pruned_weights


def prune_batchnorm(layer, channel_sparsity):
    num_features = layer.num_features
    new_num_features = int(num_features * channel_sparsity)

    layer.weight = nn.Parameter(layer.weight[:new_num_features])
    layer.bias = nn.Parameter(layer.bias[:new_num_features])
    layer.running_mean = layer.running_mean[:new_num_features]
    layer.running_var = layer.running_var[:new_num_features]
    layer.num_features = new_num_features

    return layer


def modify_resnet(model, channel_sparsity):
    first_conv = True

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            original_weights = module.weight.data

            pruned_weights = prune_weights(original_weights, channel_sparsity, is_first_layer=first_conv)
            first_conv = False

            module.out_channels = pruned_weights.shape[0]
            module.in_channels = pruned_weights.shape[1]
            module.weight = nn.Parameter(pruned_weights)

            if module.bias is not None:
                module.bias = nn.Parameter(module.bias[:module.out_channels])

        elif isinstance(module, nn.BatchNorm2d):
            prune_batchnorm(module, channel_sparsity)

    return model