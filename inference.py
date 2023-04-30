import time
import json
import os
import copy
import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np
import argparse
import cv2

from utils.module_runner import ModuleRunner
from models.model_manager import ModelManager
import utils.transforms as trans
from utils.configer import Configer

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas_Image


# activation = nn.LogSoftmax(dim=1)

def softmax(X, axis=0):
    max_prob = np.max(X, axis=axis, keepdims=True)
    X -= max_prob
    X = np.exp(X)
    sum_prob = np.sum(X, axis=axis, keepdims=True)
    X /= sum_prob
    return X

class ThermSeg():
    def __init__(self, configer):
        self.configer = configer
        self.seg_net = None
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.NormalizeThermal(norm_mode=self.configer.get('normalize', 'norm_mode')), ])
        size_mode = self.configer.get('test', 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

    def load_model(self, configer): 
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = configer.get('data', 'input_mode')
        self.seg_net.eval()

    def delete_model(self):
        if self.seg_net != None:
            del self.seg_net
            self.seg_net = None

    def run_inference(self, input_img):

        t1 = time.time()
        with torch.no_grad():
            input_img = self.img_transform(input_img)
            input_img = input_img.unsqueeze(0)
            input_img = self.module_runner.to_device(input_img)

            logits = self.seg_net.forward(input_img)
            if self.configer.get('gpu') is not None:
                torch.cuda.synchronize()
            # pred_mask = torch.argmax(activation(logits).exp(), dim=1).squeeze().cpu().numpy()
            
            logits = logits.permute(0, 2, 3, 1).cpu().numpy().squeeze()
            pred_mask = np.argmax(softmax(logits, axis=-1), axis=-1)
            time_taken = time.time() - t1

        return pred_mask, time_taken


def main(args_parser):

    # args_parser.config = 'configs/AU_GCL_RMI_Occ_High.json'

    configer = Configer(args_parser=args_parser)
    ckpt_root = configer.get('checkpoints', 'checkpoints_dir')
    ckpt_name = configer.get('checkpoints', 'checkpoints_name')
    configer.update(['network', 'resume'], os.path.join(ckpt_root, ckpt_name + '.pth'))
    thermal_file_ext = "bin"
    segObj = ThermSeg(configer)
    segObj.load_model(configer)

    img_width = 640
    img_height = 512

    datadir = args_parser.datadir
    fp_list = os.listdir(datadir)
    outdir = args_parser.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir_vis = os.path.join(args_parser.outdir, "vis")
    if not os.path.exists(outdir_vis):
        os.makedirs(outdir_vis)


    for fp in fp_list:
        fpath = os.path.join(datadir, fp)
        fp_ext = os.path.basename(fpath).split(".")[-1]
        if thermal_file_ext in fp_ext:
            print("Processing:", fp)
            try:
                thermal_matrix = np.fromfile(fpath, dtype = np.uint16, count = img_width * img_height).reshape(img_height, img_width)
                thermal_matrix = np.round(thermal_matrix  * 0.04 - 273.15, 4)    
                pred_seg_mask, time_taken = segObj.run_inference(thermal_matrix)
            except Exception as e:
                print("Error:", e)

            fp_mask = os.path.join(outdir, os.path.basename(fpath).replace(".bin", ".png"))
            cv2.imwrite(fp_mask, pred_seg_mask)

            fp_mask_vis = os.path.join(outdir_vis, os.path.basename(fpath).replace(".bin", ".jpg"))
            fig = Figure(tight_layout=True)
            canvas = FigureCanvas_Image(fig)
            ax = fig.add_subplot(111)
            if np.all(pred_seg_mask) != None:
                ax.imshow(thermal_matrix, cmap='gray')
                ax.imshow(pred_seg_mask, cmap='seismic', alpha=0.35)
            else:
                ax.imshow(thermal_matrix, cmap='magma')
            ax.set_axis_off()
            canvas.draw()
            canvas.print_jpg(fp_mask_vis)

            print("Done")


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=None,  type=str,
                        dest='datadir', help='The path containing raw thermal images.')
    parser.add_argument('--outdir', default=None,  type=str,
                        dest='outdir', help='The path to save the segmentation mask.')
    parser.add_argument('--configs', default=None,  type=str,
                        dest='configs', help='The path to congiguration file.')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()
    main(args_parser)