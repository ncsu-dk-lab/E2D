import numpy as np
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import random

def clip(image_tensor):
    """
    adjust the input based on mean and variance
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

class FastPreImgPathCache:
        def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            # Build a mapping from class index to class folder path
            self.class_folders = sorted([
                os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ])
            self.num_classes = len(self.class_folders)

        def random_img_sample(self, idx):
            class_folder = self.class_folders[idx]
            img_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            img_file = random.choice(img_files)
            img_path = os.path.join(class_folder, img_file)
            img = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return img

class PreImgPathCache(ImageFolder):
    def __init__(
            self,
            root,
            transforms,
    ):
        super(PreImgPathCache, self).__init__(root,transform=transforms)
        self.label2img = [[] for _ in range(len(self.classes))]
        for k, v in self.imgs:
            self.label2img[v].append(k)

    def random_img_sample(self,idx):
        imgpaths = self.label2img[idx]
        new_idx = np.random.choice(len(imgpaths),(1,),replace=False)[0]
        imgpath = imgpaths[new_idx]
        sample = self.loader(imgpath)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        indices = np.random.choice(list(range(len(patches))),len(patches),replace=False)
        new_patches = []
        for indice in indices:
            new_patches.append(patches[indice])
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
def tiny_clip(image_tensor):
    """
    adjust the input based on mean and variance, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor):
    """
    convert floats back to input
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def tiny_denormalize(image_tensor):
    """
    convert floats back to input, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


class ViT_BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        B, N, C = input[0].shape
        mean = torch.mean(input[0], dim=[0, 1])
        var = torch.var(input[0], dim=[0, 1], unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


# modified from Alibaba-ImageNet21K/src_files/models/utils/factory.py
def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location="cpu")

    Flag = False
    if "state_dict" in state:
        # resume from a model trained with nn.DataParallel
        state = state["state_dict"]
        Flag = True

    for key in model.state_dict():
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]

        if Flag:
            key = "module." + key

        if key in state:
            ip = state[key]
            # if key in state['state_dict']:
            #     ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print("could not load layer: {}, mismatch shape {} ,{}".format(key, (p.shape), (ip.shape)))
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model
