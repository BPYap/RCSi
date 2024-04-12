import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def _get_seed():
    return random.randint(0, 2147483647)


def _set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class RCSi:
    def __init__(self):
        self.affine_list = [
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomVerticalFlip(1.0),
            transforms.RandomAffine(degrees=0, translate=[0.1, 0.1]),
            transforms.RandomAffine(degrees=0, shear=[-15, 15, -15, 15])
        ]
        self.texture_list = [
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.Grayscale(num_output_channels=3)
            ]),
            transforms.RandomPosterize(bits=4, p=1.0),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
            transforms.RandomAutocontrast(p=1.0),
            transforms.RandomEqualize(p=1.0),
        ]


    def _sample_count(self, upper_bound):
        candidate_counts = list(range(1, upper_bound + 1))
        sampling_weights = list(range(len(candidate_counts), 0, -1))

        return random.choices(candidate_counts, weights=sampling_weights)[0]

    def _compose_transform(self):
        affine_count = self._sample_count(len(self.affine_list))
        texture_count = self._sample_count(len(self.texture_list))

        return random.sample(self.affine_list, affine_count), random.sample(self.texture_list, texture_count)

    @staticmethod
    def apply_transforms(image, transformations, random_seed=None):
        if random_seed is not None:
            _set_seeds(random_seed)
        for transform in transformations:
            image = transform(image)

        return image

    def __call__(self, image, seg_mask):
        # sample augmentations
        affine_transforms, texture_transforms = self._compose_transform()

        # apply augmentations
        affine_seed = _get_seed()
        affined_image = self.apply_transforms(image, affine_transforms, affine_seed)
        textured_image = self.apply_transforms(affined_image, texture_transforms)
        seg_mask = self.apply_transforms(seg_mask, affine_transforms, affine_seed)

        # apply self-interpolation
        lam = np.random.beta(1, 1)
        img_mask = np.uint8(np.ones((image.height, image.width)) * lam * 255)
        img_mask = Image.fromarray(img_mask)
        textured_image.paste(im=affined_image, mask=img_mask)

        return textured_image, seg_mask


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    rcsi = RCSi()
    grid_images = []
    for filename in os.listdir("samples/images"):
        img_path = os.path.join("samples/images", filename)
        mask_path = os.path.join("samples/masks", filename)
        image = Image.open(img_path) 
        mask = Image.open(mask_path)

        images = [image]
        masks = [mask]
        for _ in range(5):
            image_augmented, mask_augmented = rcsi(image, mask)
            images.append(image_augmented)
            masks.append(mask_augmented)
        grid_images.extend(images)
        grid_images.extend(masks)

    fig, axis = plt.subplots(
        nrows=14, ncols=6, figsize=(6, 11), 
        gridspec_kw={'wspace':0, 'hspace':0},
        height_ratios=[1, 1, 0.25] * 4 + [1, 1]
    )
    for row in range(14):
        for col in range(6):
            if (row + 1) % 3 != 0:
                img = grid_images.pop(0)
                img = np.asarray(img)
                axis[row, col].imshow(img)
            else:
                # display white space for every two rows
                axis[row, col].axis('off')
            axis[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
