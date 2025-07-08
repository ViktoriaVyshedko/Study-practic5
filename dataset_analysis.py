import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# Подсчёт количества изображений по классам
def count_images_per_class(dataset_path):
    class_counts = defaultdict(int)
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    return class_counts

# Анализ размеров изображений
def analyze_image_sizes(dataset_path):
    sizes = []
    size_stats = defaultdict(list)
    
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        sizes.append((width, height))
                        size_stats[class_name].append((width, height))
                except:
                    continue
    
    # Общая статистика
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    
    stats = {
        'min_width': min(widths),
        'max_width': max(widths),
        'avg_width': np.mean(widths),
        'min_height': min(heights),
        'max_height': max(heights),
        'avg_height': np.mean(heights),
        'total_images': len(sizes)
    }
    
    return stats, size_stats

# Визуализация
def visualize_stats(class_counts, size_stats):
    plt.figure(figsize=(15, 10))
    
    # Гистограмма по классам
    plt.subplot(2, 2, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Количество изображений по классам')
    plt.xticks(rotation=45)
    plt.ylabel('Количество')
    
    # Распределение размеров (ширина)
    plt.subplot(2, 2, 2)
    all_widths = [w for class_sizes in size_stats.values() for w, h in class_sizes]
    plt.hist(all_widths, bins=50, color='skyblue')
    plt.title('Распределение ширины изображений')
    plt.xlabel('Ширина (пиксели)')
    plt.ylabel('Количество')
    
    # Распределение размеров (высота)
    plt.subplot(2, 2, 3)
    all_heights = [h for class_sizes in size_stats.values() for w, h in class_sizes]
    plt.hist(all_heights, bins=50, color='salmon')
    plt.title('Распределение высоты изображений')
    plt.xlabel('Высота (пиксели)')
    plt.ylabel('Количество')
    
    # Boxplot размеров по классам
    plt.subplot(2, 2, 4)
    data = []
    labels = []
    for class_name, sizes in size_stats.items():
        if sizes:  # только классы с изображениями
            areas = [w*h for w, h in sizes]
            data.append(areas)
            labels.append(class_name)
    plt.boxplot(data, tick_labels=labels)
    plt.title('Размеры изображений по классам (площадь)')
    plt.xticks(rotation=45)
    plt.ylabel('Площадь (пиксели²)')
    
    plt.tight_layout()
    plt.show()

# Основная функция
def analyze_dataset(dataset_path):
    # Подсчёт количества изображений
    class_counts = count_images_per_class(dataset_path)
    print("Количество изображений по классам:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    
    # Анализ размеров
    stats, size_stats = analyze_image_sizes(dataset_path)
    print("\nОбщая статистика размеров:")
    print(f"Всего изображений: {stats['total_images']}")
    print(f"Ширина: min={stats['min_width']}, max={stats['max_width']}, avg={stats['avg_width']:.1f}")
    print(f"Высота: min={stats['min_height']}, max={stats['max_height']}, avg={stats['avg_height']:.1f}")
    
    # Визуализация
    visualize_stats(class_counts, size_stats)

dataset_path = "homework\data\\train"
analyze_dataset(dataset_path)