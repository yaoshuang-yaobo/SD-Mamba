import os
import sys
import cv2
import argparse
import torch
from config import cfg

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
import numpy as np
import time
from Utils.visualize import label2color
from model import get_segmentation_Model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
# model and dataset
parser.add_argument('--model', type=str, default='SD_Mamba',
                        help='model name (default: fcn32s)')
parser.add_argument('--dataset', type=str, default='cauflood',
                    help='dataset name (default: pascal_voc)')
args = parser.parse_args()


def demo(args, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)

    best_weight_path = os.path.join(cfg.DATA.WEIGHTS_PATH, best_filename)
    output_dir = os.path.join(cfg.DATA.PRE_PATH, args.model, args.dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('finish make output_dir done')

    # image transform
    transform_post = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    transform_pre = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    # input_pic = 'data/test/images/303.png'
    print('********')

    model = get_segmentation_Model(name=args.model, in_channels_pre=cfg.TRAIN.IN_CHANNELS_PRE,
                                   in_channels_post=cfg.TRAIN.IN_CHANNELS_POST,
                                   nclass=cfg.TRAIN.CLASSES).to(device)  # SD_Mamba, CDNeXt

    #model=get_segmentation_Model(name=args.model, in_channels_1=cfg.TRAIN.IN_CHANNELS_PRE,
                           #in_channels_2=cfg.TRAIN.IN_CHANNELS_POST, num_classes=cfg.TRAIN.CLASSES).to(device)  # HGINet


    print(best_weight_path)
    model_dict = torch.load(best_weight_path)
    model.load_state_dict(model_dict)

    print('Finished loading model!')

    image_path_pre = data + 'pre/'
    image_path_post = data + 'post/'
    data_list_pre = [data_index for data_index in os.listdir(image_path_pre)]
    data_list_post = [data_index for data_index in os.listdir(image_path_post)]

    for i in range(len(data_list_pre)):
        image_pre_1 = os.path.join(image_path_pre, data_list_pre[i])
        image_post_1 = os.path.join(image_path_post, data_list_post[i])

        image_pre = Image.open(image_pre_1)
        image_pre = transform_pre(image_pre).unsqueeze(0).to(device)

        image_post = Image.open(image_post_1)
        image_post = transform_post(image_post).unsqueeze(0).to(device)
        model.eval()

        with torch.no_grad():
            output, pre, post = model(image_pre, image_post)
            # print(output)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        #print(pred.shape)
        #print(pred)
        mask = label2color(cfg.TRAIN.CLASSES, pred)
        outname = os.path.splitext(os.path.split(image_pre_1)[-1])[0] + '.png'
        im = Image.fromarray(mask)
        im.save(os.path.join(output_dir, outname))
        print(os.path.join(output_dir, outname))




if __name__ == '__main__':
    data = os.path.join(cfg.DATA.IMAGE_PATH, 'test/')
    demo(args, data)

# python prediction.py --model FCN_8 --backbone vgg16 --dataset potsdam
# python prediction.py --model FCN_Mul --backbone vgg --dataset potsdam
# python prediction.py --model Fuse_FCN --backbone vgg --dataset potsdam
# python prediction.py --model Fuse_all --backbone vgg --dataset potsdam
