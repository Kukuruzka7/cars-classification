import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, df_filename, phase,
                 dataroot='../../data/intro-dl-2024',
                 img_size=320,
                 transforms=None):
        if isinstance(df_filename, str):
            self.df = pd.read_csv(df_filename)
        else:
            self.df = df_filename
        self.phase = phase
        self.dataroot = dataroot
        self.img_size = img_size
        self.transforms = transforms

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['label']

        image = cv2.imread((os.path.join(self.dataroot, self.phase, self.phase, filename)))

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image']

        image = cv2.resize(image, (self.img_size, self.img_size))

        image = self._preprocess(image)

        return image, label

    def __len__(self):
        return len(self.df)

    def _preprocess(self, image):
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5
        image = torch.from_numpy(image).float()

        return image


if __name__ == '__main__':
    train_dataset = CarDataset(df_filename='../../data/intro-dl-2024/train.csv',
                               phase='train')

    image, label = train_dataset.__getitem__(1)
    print(image.size(), label)
