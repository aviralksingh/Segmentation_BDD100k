import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a uint8 image.

    :param tensor: Input tensor (B x C x H x W) or (C x H x W) with values in range [0, 1] or any range.
    :return: NumPy array or PIL Image of the image.
    """
    # Ensure tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # If batch dimension is present (B x C x H x W), remove it
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # (C x H x W)

    # Rescale tensor to [0, 255] if necessary
    tensor = tensor.clone().detach()  # Clone to avoid modifying original tensor
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor * 255.0

    # Convert to uint8
    tensor = tensor.byte()

    # Convert (C x H x W) to (H x W x C) for image format
    np_image = tensor.permute(1, 2, 0).numpy()

    # Convert to PIL Image for easy saving or display
    return Image.fromarray(np_image)

def visualize_images(batch_size,images, labels, classes):
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(batch_size/2, batch_size/2+1))
    for i in range(batch_size):
        image= tensor_to_image(images[i])
        ax = fig.add_subplot(batch_size//2, batch_size//2+1, i+1, xticks=[], yticks=[])
        ax.imshow(image)
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, root, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    # Load random images
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    plt.show()
