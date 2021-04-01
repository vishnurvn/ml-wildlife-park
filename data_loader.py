import os
from xml.etree import ElementTree

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as t


transforms = t.Compose([
    t.ToTensor(),
    # t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

PICTURE_WIDTH = PICTURE_HEIGHT = 256
GRID = 8
grid_size = PICTURE_HEIGHT / GRID


class YoloDataset(Dataset):
    def __init__(self, path_to_annotations):
        self.path = os.path.abspath(path_to_annotations)
        self.annotations = os.listdir(self.path)

    def __getitem__(self, item):
        xml = self.annotations[item]
        path = os.path.join(self.path, xml)
        etree = ElementTree.parse(path)
        root = etree.getroot()
        tensor = torch.zeros((GRID, GRID, 3, 5))

        img_path = root.find('path').text
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transforms(img)

        for obj in root.findall('object'):
            x_min = int(obj.find('bndbox').find('xmin').text)
            y_min = int(obj.find('bndbox').find('ymin').text)
            x_max = int(obj.find('bndbox').find('xmax').text)
            y_max = int(obj.find('bndbox').find('ymax').text)

            width = x_max - x_min
            height = y_max - y_min

            x_center = x_min + width // 2
            y_center = y_min + height // 2

            x_center_ratio = x_center / PICTURE_WIDTH
            y_center_ratio = y_center / PICTURE_WIDTH
            width_ratio = width / PICTURE_WIDTH
            height_ratio = height / PICTURE_HEIGHT

            x_pos = int(x_center // grid_size)
            y_pos = int(y_center // grid_size)

            for idx in range(3):
                if tensor[..., idx, 0][y_pos][x_pos] == 0:
                    tensor[y_pos, x_pos, idx, :, ] = torch.tensor([
                        1, x_center_ratio, y_center_ratio,
                        width_ratio, height_ratio])
                    break
        return tensor, img

    def __len__(self):
        return len(self.annotations)
