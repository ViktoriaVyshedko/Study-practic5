import torch
from torchvision import transforms
from PIL import Image
from datasets import CustomImageDataset
from utils import show_images, show_single_augmentation, show_multiple_augmentations
from extra_augs import (AddGaussianNoise, RandomErasingCustom, CutOut, 
                       Solarize, Posterize, AutoContrast, ElasticTransform)
from PIL import Image, ImageFilter
import random
import torchvision.transforms.functional as F

class RandomBlur:
    def __init__(self, p=0.5, max_radius=2):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
class RandomPerspective:
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale  # степень искажения

    def __call__(self, img):
        if torch.rand(1) < self.p:
            width, height = img.size
            startpoints = [[0, 0], [width, 0], [0, height], [width, height]]
            endpoints = [
                [random.uniform(-width * self.distortion_scale, width * self.distortion_scale),
                 random.uniform(-height * self.distortion_scale, height * self.distortion_scale)],
                [width + random.uniform(-width * self.distortion_scale, width * self.distortion_scale),
                 random.uniform(-height * self.distortion_scale, height * self.distortion_scale)],
                [random.uniform(-width * self.distortion_scale, width * self.distortion_scale),
                 height + random.uniform(-height * self.distortion_scale, height * self.distortion_scale)],
                [width + random.uniform(-width * self.distortion_scale, width * self.distortion_scale),
                 height + random.uniform(-height * self.distortion_scale, height * self.distortion_scale)]
            ]
            return F.perspective(img, startpoints, endpoints)
        return img
    
class RandomBrightnessContrast:
    def __init__(self, p=0.5, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.p = p
        self.brightness_range = brightness_range  # диапазон изменения яркости
        self.contrast_range = contrast_range  # диапазон изменения контраста

    def __call__(self, img):
        if torch.rand(1) < self.p:
            brightness_factor = torch.empty(1).uniform_(*self.brightness_range).item()
            contrast_factor = torch.empty(1).uniform_(*self.contrast_range).item()
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
        return img

# Загрузка датасета без аугментаций
root = 'homework\data\\train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))

# Берем одно изображение для демонстрации
original_img, label = dataset[120]
class_names = dataset.get_class_names()
print(f"Оригинальное изображение, класс: {class_names[label]}")

# Демонстрация каждой аугментации отдельно
print("\n=== Демонстрация отдельных аугментаций ===")

# Cлучайное размытие
blur_aug = RandomBlur()
blur_img = blur_aug(original_img)
show_single_augmentation(original_img, blur_img, "Cлучайное размытие")

# Cлучайная перспектива
perspective_aug = RandomPerspective()
perspective_img = perspective_aug(original_img)
show_single_augmentation(original_img, perspective_img, "Cлучайная перспектива")

# Случайная яркость/контрастность
bright_contr_aug = transforms.Compose([
    transforms.ToTensor(),
    RandomBrightnessContrast()
])
bright_contr_img = bright_contr_aug(original_img)
show_single_augmentation(original_img, bright_contr_img, "Случайная яркость/контрастность")

# 1. Гауссов шум
noise_aug = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.2)
])
noise_img = noise_aug(original_img)
show_single_augmentation(original_img, noise_img, "Гауссов шум")

# 2. Random Erasing
erase_aug = transforms.Compose([
    transforms.ToTensor(),
    RandomErasingCustom(p=1.0)
])
erase_img = erase_aug(original_img)
show_single_augmentation(original_img, erase_img, "Random Erasing")

# 3. CutOut
cutout_aug = transforms.Compose([
    transforms.ToTensor(),
    CutOut(p=1.0, size=(32, 32))
])
cutout_img = cutout_aug(original_img)
show_single_augmentation(original_img, cutout_img, "CutOut")

# 4. Solarize
solarize_aug = transforms.Compose([
    transforms.ToTensor(),
    Solarize(threshold=128)
])
solarize_img = solarize_aug(original_img)
show_single_augmentation(original_img, solarize_img, "Solarize")

# 5. Posterize
posterize_aug = transforms.Compose([
    transforms.ToTensor(),
    Posterize(bits=4)
])
posterize_img = posterize_aug(original_img)
show_single_augmentation(original_img, posterize_img, "Posterize")

# 6. AutoContrast
autocontrast_aug = transforms.Compose([
    transforms.ToTensor(),
    AutoContrast(p=1.0)
])
autocontrast_img = autocontrast_aug(original_img)
show_single_augmentation(original_img, autocontrast_img, "AutoContrast")

# 7. Elastic Transform
elastic_aug = transforms.Compose([
    transforms.ToTensor(),
    ElasticTransform(p=1.0, alpha=1, sigma=50)
])
elastic_img = elastic_aug(original_img)
show_single_augmentation(original_img, elastic_img, "Elastic Transform")