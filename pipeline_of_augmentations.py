import os
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from typing import Dict, Callable, List
from collections import OrderedDict

class AugmentationPipeline:
    def __init__(self):
        self.augmentations = OrderedDict()
        
    def add_augmentation(self, name: str, aug: Callable) -> None:
        """Добавляет аугментацию в конвейер"""
        self.augmentations[name] = aug
        
    def remove_augmentation(self, name: str) -> None:
        """Удаляет аугментацию по имени"""
        if name in self.augmentations:
            del self.augmentations[name]
            
    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет все аугментации к изображению последовательно"""
        for aug in self.augmentations.values():
            image = aug(image)
        return image
        
    def get_augmentations(self) -> Dict[str, Callable]:
        """Возвращает словарь всех аугментаций"""
        return dict(self.augmentations)

# Реализация кастомных аугментаций
class RandomBlur:
    def __init__(self, p=0.5, max_radius=2):
        self.p = p
        self.max_radius = max_radius
        
    def __call__(self, img):
        if np.random.random() < self.p:
            radius = np.random.uniform(0.1, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class RandomPerspective:
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale
        
    def __call__(self, img):
        if np.random.random() < self.p:
            width, height = img.size
            startpoints = [(0, 0), (width, 0), (0, height), (width, height)]
            endpoints = [
                (np.random.uniform(-width*self.distortion_scale, width*self.distortion_scale),
                 np.random.uniform(-height*self.distortion_scale, height*self.distortion_scale)),
                (width + np.random.uniform(-width*self.distortion_scale, width*self.distortion_scale),
                 np.random.uniform(-height*self.distortion_scale, height*self.distortion_scale)),
                (np.random.uniform(-width*self.distortion_scale, width*self.distortion_scale),
                 height + np.random.uniform(-height*self.distortion_scale, height*self.distortion_scale)),
                (width + np.random.uniform(-width*self.distortion_scale, width*self.distortion_scale),
                 height + np.random.uniform(-height*self.distortion_scale, height*self.distortion_scale))
            ]
            return F.perspective(img, startpoints, endpoints)
        return img

# Конфигурации аугментаций
def create_light_augmentation():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.3))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(brightness=0.1, contrast=0.1))
    return pipeline

def create_medium_augmentation():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    pipeline.add_augmentation("random_rotate", transforms.RandomRotation(degrees=15))
    pipeline.add_augmentation("random_blur", RandomBlur(p=0.3, max_radius=1))
    return pipeline

def create_heavy_augmentation():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.7))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    pipeline.add_augmentation("random_rotate", transforms.RandomRotation(degrees=30))
    pipeline.add_augmentation("random_perspective", RandomPerspective(p=0.3, distortion_scale=0.5))
    pipeline.add_augmentation("random_blur", RandomBlur(p=0.5, max_radius=2))
    pipeline.add_augmentation("random_grayscale", transforms.RandomGrayscale(p=0.2))
    return pipeline

# Применение и сохранение результатов
def apply_and_save_augmentations(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    original_img = Image.open(image_path)
    
    configs = {
        "light": create_light_augmentation(),
        "medium": create_medium_augmentation(),
        "heavy": create_heavy_augmentation()
    }
    
    for config_name, pipeline in configs.items():
        # Применяем аугментации
        augmented_img = pipeline.apply(original_img)
        
        # Сохраняем результат
        output_path = os.path.join(output_dir, f"{config_name}_{os.path.basename(image_path)}")
        augmented_img.save(output_path)
        
        print(f"Saved {config_name} augmentation to {output_path}")

# Пример использования
if __name__ == "__main__":
    # Применяем ко всем изображениям в train директории
    train_dir = "path/to/train"
    output_base = "path/to/augmented_train"
    
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                output_dir = os.path.join(output_base, class_dir)
                apply_and_save_augmentations(img_path, output_dir)