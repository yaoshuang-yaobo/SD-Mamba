
import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from config import cfg
import torchvision.transforms as T


class Wuhan(Dataset):

    def __init__(self, split, height=256, width=256, classes=2):
        self.classes = classes
        self.height, self.width = height, width
        if split == 'train':
            self.data_path = os.path.join(cfg.DATA.IMAGE_PATH, 'train/')
        elif split == 'val':
            self.data_path = os.path.join(cfg.DATA.IMAGE_PATH, 'val/')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))
        self.image_path_pre = self.data_path + 'pre/'
        self.image_path_post = self.data_path + 'post/'
        self.label_path = self.data_path + 'labels/'
        self.data_list_pre = [data_index for data_index in os.listdir(self.image_path_pre)]
        self.data_list_post = [data_index for data_index in os.listdir(self.image_path_post)]

    def _read_data_path(self, i):
        image_pre = os.path.join(self.image_path_pre, self.data_list_pre[i])
        image_post = os.path.join(self.image_path_post, self.data_list_post[i])
        label = os.path.join(self.label_path, self.data_list_pre[i])
        return image_pre, image_post, label

    @classmethod
    def _SegColor2Label(cls, img):
        """
        img: Shape [h, w, 3]
        mapMatrix: color-> label mapping matrix,
                   覆盖了Uint8 RGB空间所有256x256x256种颜色对应的label

        return: labelMatrix: Shape [h, w], 像素值即类别标签
        """
        VOC_COLORMAP = [[255, 255, 255], [0, 0, 0]]
        # 水：[255, 255, 255]
        # 背景：[0, 0, 0]
        mapMatrix = np.zeros(256*256*256, dtype=np.int32)
        for i, cm in enumerate(VOC_COLORMAP):
            mapMatrix[cm[0] * 65536 + cm[1] * 256 + cm[2]] = i

        indices = img[:, :, 0] * 65536 + img[:, :, 1] * 256 + img[:, :, 2]
        return mapMatrix[indices]

    def _get_labels(self, path):
        """
        :param path:
        :return:
        """
        # Load labels
        seg_labels = np.zeros((self.height, self.width))
        try:
            #print(path)
            labels = Image.open(path) # PIL类型图像需要转成torch需要的numpy类型就是（b, c, n, w）
            labels = np.array(labels, dtype=np.uint8)  # 转成numpy类型
            seg_labels = self._SegColor2Label(labels)

        except Exception:
            print('标签读取或者转换错误：{}'.format(Exception))
        return seg_labels

    def _img_transforms_pre(self, data):
        im_tfs = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256))
        ])
        data = im_tfs(data)
        return data

    def _img_transforms_post(self, data):
        im_tfs = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Resize((256, 256))
        ])
        data = im_tfs(data)
        return data

    # 进行切片
    def __getitem__(self, item):
        image_path_pre, image_path_post, label_path = self._read_data_path(item)
        image_pre = Image.open(image_path_pre)
        image_pre = self._img_transforms_pre(image_pre)

        image_post = Image.open(image_path_post)
        image_post = self._img_transforms_post(image_post)

        mask = self._get_labels(label_path)
        mask = torch.from_numpy(mask).long()

        return image_pre, image_post, mask

    def __len__(self):
        return len(self.data_list_pre)


if __name__ == '__main__':

    data = Wuhan('val', 256, 256, 2)
    for img_pre, img_post, label in data:
        print(label.shape)