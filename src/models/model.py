import torch.nn as nn
import torchvision.models as models
from torchvision.models._api import WeightsEnum


def build_model(
    model_name: str,
    num_classes: int = 1,
    weights: WeightsEnum | None = None,
    in_channels: int = 3,
) -> nn.Module:
    """
    Build a model by name with configurable input/output layers.

    Args:
        model_name (str): The name of the model to build (e.g., 'resnet18', 'vgg16').
        num_classes (int): The number of output classes.
        weights (WeightsEnum | None): The pretrained weights to use.
        in_channels (int): The number of input channels.

    Returns:
        nn.Module: The configured model.
    """
    model_name = model_name.lower()
    if model_name not in models.__dict__:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

    model = models.__dict__[model_name](weights=weights)

    # Adjust the first convolutional layer if input channels differ from the default
    if in_channels != 3:
        first_conv_layer = _get_first_conv(model)
        if first_conv_layer:
            new_conv = nn.Conv2d(
                in_channels,
                first_conv_layer.out_channels,
                kernel_size=first_conv_layer.kernel_size,
                stride=first_conv_layer.stride,
                padding=first_conv_layer.padding,
                bias=first_conv_layer.bias,
            )
            _replace_layer(model, first_conv_layer, new_conv)

    # Adjust the final classification layer
    if hasattr(model, "fc"):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, "classifier"):
        # For models with a classifier, replace the last layer of the Sequential module
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError(
            f"The model '{model_name}' is not supported for automatic final layer modification."
        )

    return model


def _get_first_conv(model: nn.Module) -> nn.Module | None:
    """Helper to find the first convolutional layer of a model."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None


def _replace_layer(model: nn.Module, old_layer: nn.Module, new_layer: nn.Module):
    """Helper to replace a layer in a model by iterating over named children."""
    for name, module in model.named_children():
        if module is old_layer:
            setattr(model, name, new_layer)
            return
