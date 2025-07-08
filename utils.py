import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original_img, augmented_img, title):
    """Универсальная функция визуализации для PIL, тензоров и numpy массивов"""
    
    def convert_to_display(img):
        # Если это PIL Image
        if isinstance(img, Image.Image):
            img = np.array(img)
            # Если изображение RGBA, конвертируем в RGB
            if img.shape[-1] == 4:
                img = img[..., :3]
            return img
        
        # Если это PyTorch тензор
        if hasattr(img, 'numpy'):
            img = img.numpy()
            # Меняем порядок каналов (C,H,W) -> (H,W,C)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
            return img
        
        # Если это numpy массив
        if isinstance(img, np.ndarray):
            # Нормализуем, если значения выходят за пределы [0,1]
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() > 1.0:
                    img = img / 255.0
            return img
        
        raise TypeError(f"Неподдерживаемый тип изображения: {type(img)}")
    
    # Конвертируем изображения
    orig_np = convert_to_display(original_img)
    aug_np = convert_to_display(augmented_img)
    
    # Проверяем типы данных
    if orig_np.dtype == np.object_ or aug_np.dtype == np.object_:
        raise ValueError("Обнаружены данные типа object. Проверьте входные изображения.")
    
    # Создаём график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Отображаем оригинал
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Отображаем аугментированное изображение
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles):
    # Преобразуем изображение в numpy и меняем оси (C, H, W) → (H, W, C)
    if hasattr(original_img, 'numpy'):  # Если это тензор PyTorch
        orig_np = original_img.numpy().transpose(1, 2, 0)
    else:  # Если это PIL.Image или уже numpy-массив
        orig_np = np.array(original_img)
        if orig_np.ndim == 3 and orig_np.shape[0] == 3:  # Если каналы на первом месте
            orig_np = orig_np.transpose(1, 2, 0)  # Меняем оси

    # Создаем subplot
    fig, axes = plt.subplots(1, len(augmented_imgs) + 1, figsize=(15, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Отображаем аугментированные изображения
    for i, (img, title) in enumerate(zip(augmented_imgs, titles), 1):
        # Аналогично обрабатываем аугментированные изображения
        if hasattr(img, 'numpy'):
            img_np = img.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
        
        axes[i].imshow(img_np)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.show()