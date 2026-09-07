"""
Dataset loaders for StruFreqDiT (binary medical image segmentation).

Expected directory layout, relative to the project root (see README.md):

  processed_glas/ | processed_monuseg/ | processed_ph2/ | processed_imid/
    train/images/*.png   train/masks/*.png
    val/images/*.png     val/masks/*.png
    test/images/*.png    test/masks/*.png

An image and its mask are matched by file name stem, so their extensions may
differ. Ranges after loading:
    image -> (3, H, W) float in [-1, 1]
    mask  -> (1, H, W) float in {-1, +1}
"""
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# Dataset root directories (relative to the project root)
_DATASET_ROOT = {
    'glas':    'processed_glas',
    'monuseg': 'processed_monuseg',
    'ph2':     'processed_ph2',
    'imid':    'processed_imid',
    # Evaluation only: target domain of the MoNuSeg -> TNBC transfer experiment
    'tnbc':    'processed_tnbc',
}


def get_dataset_root(dataset_name: str) -> str:
    root = _DATASET_ROOT.get(dataset_name.lower())
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset_name}, "
                         f"supported: {list(_DATASET_ROOT)}")
    return root


class MedicalSegmentationDataset(Dataset):
    """Binary medical image segmentation dataset (image + mask pairs)."""

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256):
        """
        Args:
            root_dir  : dataset root, e.g. 'processed_glas'
            split     : 'train' | 'val' | 'test'
            image_size: images and masks are resized to image_size x image_size
        """
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.is_train   = (split == 'train')

        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir  = os.path.join(root_dir, split, 'masks')

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_dir}\n"
                f"Expected {root_dir}/{{train,val,test}}/{{images,masks}}/ "
                f"- see the 'Data' section of README.md"
            )

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        if len(self.filenames) == 0:
            raise RuntimeError(f"Empty dataset: {self.image_dir}")

        # Match each image to its mask by file name stem
        _exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        mask_by_stem = {
            os.path.splitext(m)[0]: m
            for m in os.listdir(self.mask_dir) if m.lower().endswith(_exts)
        }
        self.mask_filenames = []
        for f in self.filenames:
            stem = os.path.splitext(f)[0]
            if stem not in mask_by_stem:
                raise FileNotFoundError(
                    f"No mask found for image {f} in {self.mask_dir} (stem '{stem}')"
                )
            self.mask_filenames.append(mask_by_stem[stem])

        self.resize      = transforms.Resize((image_size, image_size))
        self.resize_mask = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def __len__(self):
        return len(self.filenames)

    def apply_augmentation(self, image: Image.Image, mask: Image.Image):
        """Synchronized augmentation for image and mask (train split only)."""
        # 1. Random 90-degree rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        # 2. Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 3. Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # 4. Random scale + crop (0.8-1.2x)
        if random.random() > 0.3:
            scale    = random.uniform(0.8, 1.2)
            new_size = int(self.image_size * scale)
            image = TF.resize(image, new_size)
            mask  = TF.resize(mask,  new_size,
                              interpolation=transforms.InterpolationMode.NEAREST)
            if scale > 1.0:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, (self.image_size, self.image_size))
                image = TF.crop(image, i, j, h, w)
                mask  = TF.crop(mask,  i, j, h, w)
            else:
                image = TF.resize(image, self.image_size)
                mask  = TF.resize(mask,  self.image_size,
                                  interpolation=transforms.InterpolationMode.NEAREST)

        # 5. Color jitter (image only)
        if random.random() > 0.3:
            image = self.color_jitter(image)

        # 6. Random Gaussian blur (image only)
        if random.random() > 0.7:
            image = TF.gaussian_blur(image, random.choice([3, 5]))

        return image, mask

    def __getitem__(self, idx):
        fname     = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir,  self.mask_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        image = self.resize(image)
        mask  = self.resize_mask(mask)

        if self.is_train:
            image, mask = self.apply_augmentation(image, mask)

        image = self.to_tensor(image)          # (3, H, W) in [0, 1]
        mask  = self.to_tensor(mask)           # (1, H, W) in [0, 1]

        return {
            'image':    image * 2.0 - 1.0,                    # [-1, +1]
            'mask':     (mask > 0.5).float() * 2.0 - 1.0,     # {-1, +1}
            'filename': fname,
        }


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_dataloader(dataset_name: str, batch_size: int = 8, image_size: int = 256,
                   num_workers: int = 4):
    """Train and test DataLoaders. Returns (train_loader, test_loader)."""
    root = get_dataset_root(dataset_name)
    return (
        _make_loader(MedicalSegmentationDataset(root, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        _make_loader(MedicalSegmentationDataset(root, 'test', image_size),
                     batch_size, False, num_workers),
    )


def get_dataloader_with_val(dataset_name: str, batch_size: int = 8,
                            image_size: int = 256, num_workers: int = 4):
    """Train / val / test DataLoaders (splits are fixed on disk).

    Returns (train_loader, val_loader, test_loader).
    """
    root = get_dataset_root(dataset_name)
    return (
        _make_loader(MedicalSegmentationDataset(root, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        _make_loader(MedicalSegmentationDataset(root, 'val', image_size),
                     batch_size, False, num_workers),
        _make_loader(MedicalSegmentationDataset(root, 'test', image_size),
                     batch_size, False, num_workers),
    )
