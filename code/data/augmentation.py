import torch
import torch.nn.functional as F


def cutout(batch):
    length, replace = 16, 0.0
    images, labels = batch
    num_channels = images.shape[1]
    image_height, image_width = images.shape[2], images.shape[3]
    cutout_center_height = torch.randint(0, image_height, (1,)).item()
    cutout_center_width = torch.randint(0, image_width, (1,)).item()
    lower_pad = max(0, cutout_center_height - length // 2)
    upper_pad = max(0, image_height - cutout_center_height - length // 2)
    left_pad = max(0, cutout_center_width - length // 2)
    right_pad = max(0, image_width - cutout_center_width - length // 2)
    cutout_shape = [
    image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)
    ]
    mask = F.pad(torch.zeros(cutout_shape, dtype=images.dtype), (left_pad, right_pad, upper_pad, lower_pad), value=1)[..., None]
    patch = torch.ones_like(images, dtype=images.dtype) * replace
    mask = torch.tile(mask, (1, 1, num_channels)).permute(2, 0, 1)
    images = torch.where(mask == 0, patch, images)
    return [images, labels]
