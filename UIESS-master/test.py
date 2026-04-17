import argparse
import os
import sys
import time
from PIL import Image

import numpy as np
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import *
from datasets import *

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="release",
                    help="sub-folder name matching saved_models/<exp_name>/")
parser.add_argument("--test_dir", type=str,
                    default='/kaggle/input/uieb-dataset/UIESS-master/data/testA',
                    help="directory of input images to enhance")
parser.add_argument("--out_dir", type=str,
                    default='/kaggle/working/output',
                    help="root output directory")
parser.add_argument("--model_dir", type=str,
                    default='/kaggle/input/uieb-dataset/UIESS-master/saved_models/release',
                    help="directory containing *.pth weight files")
parser.add_argument("--data_root", type=str,
                    default='/kaggle/input/uieb-dataset/UIESS-master/data',
                    help="dataset root (needed for test_plot_latent_tsne / test_samples)")
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--checkpoint", type=int, default=29,
                    help="epoch number of the checkpoint to load")
parser.add_argument("--style_dim", type=int, default=8)
parser.add_argument("--num_sample", type=int, default=2)
parser.add_argument("--n_residual", type=int, default=3)
parser.add_argument("--dim", type=int, default=40)
parser.add_argument("--n_downsample", type=int, default=2)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--print_model_complexity", type=bool, default=True)
parser.add_argument("--seed", type=int, default=123)

# Jupyter / ipykernel passes extra args → ignore them safely
if any("ipykernel" in a or "jupyter" in a for a in sys.argv):
    opt = parser.parse_args(args=[])
else:
    opt = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True


# ── Test functions ────────────────────────────────────────────────────────────

def test_REAL_image(epoch):
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_REAL_image', str(epoch))
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [transforms.ToTensor()]
    val_dataloader = DataLoader(
        EnhancedValDataset(transforms_=transforms_val, dataset_path=opt.test_dir),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if opt.print_model_complexity:
        num_params = sum(p.numel()
                         for m in [c_Enc, real_sty_Enc, G, T]
                         for p in m.parameters())
        print('Total number of parameters: %d' % num_params)

    time_all = 0
    for i, batch in enumerate(val_dataloader):
        imgReal = batch["img"].to(device)
        name    = batch["name"][0].split(os.sep)[-1]
        with torch.no_grad():
            start = time.time()

            c_code_Real  = c_Enc(imgReal)
            s_code_Real  = real_sty_Enc(imgReal)
            en_s_code    = T(s_code_Real)
            enhanced     = G(c_code_Real, en_s_code)

            if opt.print_model_complexity and i != 0:
                time_all += time.time() - start

            ndarr = (enhanced.squeeze()
                              .mul(255).add_(0.5).clamp_(0, 255)
                              .permute(1, 2, 0)
                              .to('cpu', torch.uint8).numpy())
            im = Image.fromarray(ndarr)

            ori_im = Image.open(batch["name"][0])
            im = im.resize(ori_im.size)
            im.save(os.path.join(out_path, name))

    if len(val_dataloader) > 1:
        print("Total time: %f, average time: %f, FPS: %f, dataloader: %d" % (
            time_all,
            time_all / max(len(val_dataloader) - 1, 1),
            max(len(val_dataloader) - 1, 1) / max(time_all, 1e-9),
            len(val_dataloader)))


def test_SYN_image():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_SYN_image')
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [transforms.ToTensor()]
    val_dataloader = DataLoader(
        EnhancedValDataset(transforms_=transforms_val, dataset_path=opt.test_dir),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if opt.print_model_complexity:
        num_params = sum(p.numel()
                         for m in [c_Enc, syn_sty_Enc, G, T]
                         for p in m.parameters())
        print('Total number of parameters: %d' % num_params)

    time_all = 0
    for i, batch in enumerate(val_dataloader):
        imgSyn = batch["img"].to(device)
        name   = batch["name"][0].split(os.sep)[-1]
        with torch.no_grad():
            start = time.time()

            c_code_Syn  = c_Enc(imgSyn)
            s_code_Syn  = syn_sty_Enc(imgSyn)
            en_s_code   = T(s_code_Syn)
            enhanced    = G(c_code_Syn, en_s_code)

            if opt.print_model_complexity and i != 0:
                time_all += time.time() - start
                print("time: ", time.time() - start)

            ndarr = (enhanced.squeeze()
                              .mul(255).add_(0.5).clamp_(0, 255)
                              .permute(1, 2, 0)
                              .to('cpu', torch.uint8).numpy())
            im = Image.fromarray(ndarr)

            ori_im = Image.open(batch["name"][0])
            im = im.resize(ori_im.size)
            im.save(os.path.join(out_path, name))

    if len(val_dataloader) > 1:
        print("Total time: %f, average time: %f, FPS: %f, dataloader: %d" % (
            time_all,
            time_all / max(len(val_dataloader) - 1, 1),
            max(len(val_dataloader) - 1, 1) / max(time_all, 1e-9),
            len(val_dataloader)))


def test_plot_latent_tsne():
    transforms_val = [transforms.ToTensor()]
    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root, transforms_=transforms_val, mode="val"),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    feature, label = [], []
    for i, batch in enumerate(val_dataloader):
        imgReal = batch["Real"].to(device)
        imgSyn  = batch["Syn"].to(device)

        with torch.no_grad():
            s_code_Real  = real_sty_Enc(imgReal)
            en_s_code_Real = T(s_code_Real)

            s_code_Syn   = syn_sty_Enc(imgSyn)
            en_s_code_Syn = T(s_code_Syn)

            feature.append(s_code_Real.cpu().detach().numpy().squeeze())
            label.append('syn')
            feature.append(s_code_Syn.cpu().detach().numpy().squeeze())
            label.append('real-world')
            feature.append(en_s_code_Real.cpu().detach().numpy().squeeze())
            label.append('clean')
            feature.append(en_s_code_Syn.cpu().detach().numpy().squeeze())
            label.append('clean')

    feature = np.array(feature)
    label   = np.array(label)
    X = TSNE(perplexity=18.0, learning_rate='auto',
             random_state=123, verbose=1).fit_transform(feature)
    sns.scatterplot(X[:, 0], X[:, 1], hue=label, legend='full', palette='Set2')
    plt.show()


def test_latent_manipulation():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_latent_manipulation')
    os.makedirs(out_path, exist_ok=True)

    alphas = np.array([0, 0.25, 0.5, 0.7, 0.8, 1.0, 1.33, 1.5, 1.7])

    transforms_val = transforms.Compose([transforms.ToTensor()])

    img_list = os.listdir(opt.test_dir)
    for img_name in img_list:
        img_path = os.path.join(opt.test_dir, img_name)
        img = load_img(img_path)
        w, h = img.size
        new_w = w // 4 * 4 if (w / 4) % 1 != 0 else w
        new_h = h // 4 * 4 if (h / 4) % 1 != 0 else h
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        img = transforms_val(img).to(device).unsqueeze(0)
        item_list  = []
        img_enhanceds = None

        with torch.no_grad():
            c_code, s_code_ori = c_Enc(img), real_sty_Enc(img)
            en_s_code = T(s_code_ori)

            for alpha in alphas:
                s_code  = s_code_ori + alpha * (en_s_code - s_code_ori)
                enhanced = G(c_code, s_code)
                item_list.append(enhanced)

            img_out = None
            for item in item_list:
                img_out = item if img_out is None else torch.cat((img_out, item), -1)

            img_enhanceds = img_out
            save_image(img_enhanceds,
                       os.path.join(out_path, img_name),
                       nrow=1, normalize=True, value_range=(0, 1))
            print(img_name)


def test_samples():
    out_path = os.path.join(opt.out_dir, opt.exp_name, 'test_samples')
    os.makedirs(out_path, exist_ok=True)

    transforms_val = [transforms.ToTensor()]
    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root, transforms_=transforms_val, mode="val"),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    def sample_images(batches_done):
        for i, batch in enumerate(val_dataloader):
            img_enhanceds_Real = None
            img_enhanceds_Syn  = None
            for imgReal, imgSyn, label_Syn in zip(
                    batch["Real"], batch["Syn"], batch["label"]):
                with torch.no_grad():
                    XReal = imgReal.unsqueeze(0).to(device)
                    XSyn  = imgSyn.unsqueeze(0).to(device)

                    c_code_Real, s_code_Real = c_Enc(XReal), real_sty_Enc(XReal)
                    c_code_Syn,  s_code_Syn  = c_Enc(XSyn),  syn_sty_Enc(XSyn)

                    XRealSyn = G(c_code_Real, s_code_Syn)
                    XSynReal = G(c_code_Syn,  s_code_Real)

                    en_s_code_Real = T(s_code_Real)
                    en_s_code_Syn  = T(s_code_Syn)

                    enhanced_Real = G(c_code_Real, en_s_code_Real)
                    enhanced_Syn  = G(c_code_Syn,  en_s_code_Syn)

                    c_code_SynReal, s_code_SynReal = c_Enc(XSynReal), real_sty_Enc(XSynReal)
                    c_code_RealSyn, s_code_RealSyn = c_Enc(XRealSyn), syn_sty_Enc(XRealSyn)
                    XRealSynReal = G(c_code_RealSyn, s_code_Real)
                    XSynRealSyn  = G(c_code_SynReal, s_code_Syn)

                    XRealReal = G(c_code_Real, s_code_Real)
                    XSynSyn   = G(c_code_Syn,  s_code_Syn)

                    item_list = [XSynSyn, XSynReal, XSynRealSyn, enhanced_Syn,
                                 label_Syn.to(device).unsqueeze(0)]
                    imgSyn = imgSyn.to(device).unsqueeze(0)
                    for item in item_list:
                        imgSyn = torch.cat((imgSyn, item), -1)

                    item_list = [XRealReal, XRealSyn, XRealSynReal, enhanced_Real]
                    imgReal = imgReal.to(device).unsqueeze(0)
                    for item in item_list:
                        imgReal = torch.cat((imgReal, item), -1)

                    img_enhanceds_Real = (imgReal if img_enhanceds_Real is None
                                         else torch.cat((img_enhanceds_Real, imgReal), -2))
                    img_enhanceds_Syn  = (imgSyn  if img_enhanceds_Syn  is None
                                         else torch.cat((img_enhanceds_Syn,  imgSyn),  -2))

            save_image(img_enhanceds_Real,
                       os.path.join(out_path, "%s_I2I_Enhanced_Real.png" % str(i)),
                       nrow=1, normalize=True, value_range=(0, 1))
            save_image(img_enhanceds_Syn,
                       os.path.join(out_path, "%s_I2I_Enhanced_Syn.png" % str(i)),
                       nrow=1, normalize=True, value_range=(0, 1))
            if i > batches_done:
                return

    sample_images(batches_done=300)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    testing = 'test_REAL_image'
    # Options: test_REAL_image | test_SYN_image | test_plot_latent_tsne
    #          test_latent_manipulation | test_samples

    # ── Build models ──────────────────────────────────────────────────────────
    c_Enc        = ContentEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                  n_residual=opt.n_residual).to(device)
    G            = Generator(dim=opt.dim, n_upsample=opt.n_downsample,
                             n_residual=opt.n_residual,
                             style_dim=opt.style_dim).to(device)
    real_sty_Enc = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                style_dim=opt.style_dim).to(device)
    syn_sty_Enc  = StyleEncoder(dim=opt.dim, n_downsample=opt.n_downsample,
                                style_dim=opt.style_dim).to(device)
    T            = StyleTransformUnit(dim=opt.dim,
                                      style_dim=opt.style_dim).to(device)

    # ── Load weights ──────────────────────────────────────────────────────────
    ckpt = opt.checkpoint
    c_Enc.load_state_dict(
        torch.load(os.path.join(opt.model_dir, "c_Enc_%d.pth" % ckpt),
                   map_location=device))
    G.load_state_dict(
        torch.load(os.path.join(opt.model_dir, "G_%d.pth" % ckpt),
                   map_location=device))
    real_sty_Enc.load_state_dict(
        torch.load(os.path.join(opt.model_dir, "real_sty_Enc_%d.pth" % ckpt),
                   map_location=device))
    syn_sty_Enc.load_state_dict(
        torch.load(os.path.join(opt.model_dir, "syn_sty_Enc_%d.pth" % ckpt),
                   map_location=device))
    T.load_state_dict(
        torch.load(os.path.join(opt.model_dir, "T_%d.pth" % ckpt),
                   map_location=device))

    # Set eval mode
    for m in [c_Enc, G, real_sty_Enc, syn_sty_Enc, T]:
        m.eval()

    # ── Run ───────────────────────────────────────────────────────────────────
    if testing == 'test_REAL_image':
        test_REAL_image(epoch=ckpt)
    elif testing == 'test_SYN_image':
        test_SYN_image()
    elif testing == 'test_plot_latent_tsne':
        test_plot_latent_tsne()
    elif testing == 'test_latent_manipulation':
        test_latent_manipulation()
    elif testing == 'test_samples':
        test_samples()