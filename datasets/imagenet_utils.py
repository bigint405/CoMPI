import os
import json
from torch.utils.data import Dataset
from PIL import Image


def get_all_catagories():
    # 指定目录
    with open('configs/dataset_configs.json', 'r') as f:
        data = json.load(f)
    directory = os.path.join(data['path'], 'val')

    # 获取目录下的文件夹名称，并存储到列表中
    return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

def get_train_dir():
    with open('configs/dataset_configs.json', 'r') as f:
        data = json.load(f)
    return os.path.join(data['path'], 'train')

def get_val_dir():
    with open('configs/dataset_configs.json', 'r') as f:
        data = json.load(f)
    return os.path.join(data['path'], 'val')

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, target_classes, transform=None, preload=False):
        self.root_dir = root_dir
        self.target_classes = target_classes
        self.transform = transform
        self.preload = preload
        self.loaded = False

        self.samples = []     # [(img_path, label)]
        self.images = []      # if preloaded: [PIL image]
        self.labels = []
        self.idx_map = {}
        self.target_count = 0

        self._build_index()
        if self.preload:
            self.load()

    def _build_index(self):
        """构建文件路径和标签索引，不加载图像"""
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            if class_name in self.target_classes:
                class_path = os.path.join(self.root_dir, class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    label = self.idx_map.setdefault(idx, self.target_count)
                    if label == self.target_count:
                        self.target_count += 1
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload and not self.loaded:
            self.load()

        if self.preload:
            image = self.images[idx]
        else:
            image_path, _ = self.samples[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

        label = self.labels[idx] if self.preload else self.samples[idx][1]
        return image, label

    def load(self):
        if not self.preload:
            return
        if not self.loaded:
            self.images = []
            self.labels = []
            for img_path, label in self.samples:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.images.append(image)
                self.labels.append(label)
            self.loaded = True

    def unload(self):
        if not self.preload:
            return
        """手动释放内存（如果预加载过）"""
        if self.loaded:
            self.images = []
            self.labels = []
            self.loaded = False