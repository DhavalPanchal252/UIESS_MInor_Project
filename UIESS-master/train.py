import argparse
import datetime
import itertools
import time
import sys
import os
import random

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

from datasets import *
from loss import SSIM, VGGPerceptualLoss, TVLoss, DCPLoss
from models import *

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str,
                    default='/kaggle/input/uieb-dataset/UIESS-master/data',
                    help="dataset root (trainA / trainB / trainB_label inside)")
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=35,
                    help="number of training epochs")
parser.add_argument("--exp_name", type=str, default="Refactor testing",
                    help="experiment name (used for save paths)")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--decay_epoch", type=int, default=20)
parser.add_argument("--n_cpu", type=int, default=0,
                    help="DataLoader workers (keep 0 on Kaggle)")
parser.add_argument("--img_height", type=int, default=128)
parser.add_argument("--img_width", type=int, default=128)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--sample_interval", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=10)
parser.add_argument("--n_downsample", type=int, default=2)
parser.add_argument("--n_residual", type=int, default=3)
parser.add_argument("--dim", type=int, default=40)
parser.add_argument("--style_dim", type=int, default=8)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out_dir", type=str, default='/kaggle/working/output',
                    help="root directory for all saved images and models")

# Jupyter / ipykernel passes extra args that argparse can't parse → ignore them
if any("ipykernel" in a or "jupyter" in a for a in sys.argv):
    opt = parser.parse_args(args=[])
else:
    opt = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    images_dir = os.path.join(opt.out_dir, "images", opt.exp_name)
    models_dir = os.path.join(opt.out_dir, "saved_models", opt.exp_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    set_seed(opt.seed)

    # ── Losses ────────────────────────────────────────────────────────────────
    criterion_recon = torch.nn.L1Loss().to(device)
    ssim_loss       = SSIM().to(device)
    tv_loss         = TVLoss(1).to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    # ── Models ────────────────────────────────────────────────────────────────
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
    D            = MultiDiscriminator().to(device)

    if opt.epoch != 0:
        # Resume training – load checkpoints
        c_Enc.load_state_dict(
            torch.load(os.path.join(models_dir, "c_Enc_%d.pth" % opt.epoch),
                       map_location=device))
        G.load_state_dict(
            torch.load(os.path.join(models_dir, "G_%d.pth" % opt.epoch),
                       map_location=device))
        real_sty_Enc.load_state_dict(
            torch.load(os.path.join(models_dir, "real_sty_Enc_%d.pth" % opt.epoch),
                       map_location=device))
        syn_sty_Enc.load_state_dict(
            torch.load(os.path.join(models_dir, "syn_sty_Enc_%d.pth" % opt.epoch),
                       map_location=device))
        T.load_state_dict(
            torch.load(os.path.join(models_dir, "T_%d.pth" % opt.epoch),
                       map_location=device))
        D.load_state_dict(
            torch.load(os.path.join(models_dir, "D_%d.pth" % opt.epoch),
                       map_location=device))
    else:
        c_Enc.apply(weights_init_normal)
        G.apply(weights_init_normal)
        real_sty_Enc.apply(weights_init_normal)
        syn_sty_Enc.apply(weights_init_normal)
        T.apply(weights_init_normal)
        D.apply(weights_init_normal)

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_gan              = 1
    lambda_id               = 10
    lambda_cyc              = 1
    lambda_enhanced         = 3.5 / 2
    lambda_ssim             = 5.0 / 2
    lambda_tv               = 0.3
    lambda_perceptual       = 0.0005 / 2
    lambda_enhanced_latent  = 3

    # ── Optimizers ────────────────────────────────────────────────────────────
    optimizer_G = torch.optim.Adam(
        itertools.chain(c_Enc.parameters(), G.parameters(),
                        real_sty_Enc.parameters(), syn_sty_Enc.parameters(),
                        T.parameters()),
        lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(
        D.parameters(), lr=opt.lr * 5, betas=(opt.b1, opt.b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D1,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # ── Data loaders ──────────────────────────────────────────────────────────
    transforms_train = [transforms.ToTensor()]
    transforms_val   = [transforms.Resize((128, 128)), transforms.ToTensor()]

    set_seed(opt.seed)
    dataloader = DataLoader(
        EnhancedDataset(opt.data_root, transforms_=transforms_train),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,          # must be 0 on Kaggle
        worker_init_fn=worker_init,
        pin_memory=True)

    set_seed(opt.seed)
    val_dataloader = DataLoader(
        EnhancedDataset(opt.data_root, transforms_=transforms_val, mode="val"),
        batch_size=5,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True)

    # ── Sample helper ─────────────────────────────────────────────────────────
    def sample_images(batches_done):
        """Save a generated sample from the validation set."""
        imgs = next(iter(val_dataloader))

        img_enhanceds_A = None
        img_enhanceds_B = None

        for imgA, imgB, label_B in zip(imgs["Real"], imgs["Syn"], imgs["label"]):
            with torch.no_grad():
                XA = imgA.unsqueeze(0).to(device)
                XB = imgB.unsqueeze(0).to(device)

                c_code_A, s_code_A = c_Enc(XA), real_sty_Enc(XA)
                c_code_B, s_code_B = c_Enc(XB), syn_sty_Enc(XB)

                XAB = G(c_code_A, s_code_B)
                XBA = G(c_code_B, s_code_A)

                en_s_code_A = T(s_code_A)
                en_s_code_B = T(s_code_B)

                enhanced_A = G(c_code_A, en_s_code_A)
                enhanced_B = G(c_code_B, en_s_code_B)

                c_code_BA, s_code_BA = c_Enc(XBA), real_sty_Enc(XBA)
                c_code_AB, s_code_AB = c_Enc(XAB), syn_sty_Enc(XAB)
                XABA = G(c_code_AB, s_code_A) if lambda_cyc > 0 else 0
                XBAB = G(c_code_BA, s_code_B) if lambda_cyc > 0 else 0

                XAA = G(c_code_A, s_code_A)
                XBB = G(c_code_B, s_code_B)

                item_list = [XBB, XBA, XBAB, enhanced_B, label_B.unsqueeze(0).to(device)]
                imgB = imgB.unsqueeze(0).to(device)
                for item in item_list:
                    imgB = torch.cat((imgB, item), -1)

                item_list = [XAA, XAB, XABA, enhanced_A]
                imgA = imgA.unsqueeze(0).to(device)
                for item in item_list:
                    imgA = torch.cat((imgA, item), -1)

                img_enhanceds_A = imgA if img_enhanceds_A is None else torch.cat(
                    (img_enhanceds_A, imgA), -2)
                img_enhanceds_B = imgB if img_enhanceds_B is None else torch.cat(
                    (img_enhanceds_B, imgB), -2)

        save_image(img_enhanceds_A,
                   os.path.join(images_dir, "%s_I2I_Enhanced_A.png" % batches_done),
                   nrow=5, normalize=True)
        save_image(img_enhanceds_B,
                   os.path.join(images_dir, "%s_I2I_Enhanced_B.png" % batches_done),
                   nrow=5, normalize=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    valid = 1
    fake  = 0

    prev_time = time.time()
    sample_images(0)

    for epoch in range(opt.epoch + 1 if opt.epoch > 0 else 0, opt.n_epochs + 1):
        for i, batch in enumerate(dataloader):
            optimizer_G.zero_grad()
            optimizer_D1.zero_grad()

            XA     = batch["Real"].to(device)
            XB     = batch["Syn"].to(device)
            labelB = batch["label"].to(device)

            # ── Encode ────────────────────────────────────────────────────────
            c_code_A, s_code_A = c_Enc(XA), real_sty_Enc(XA)
            c_code_B, s_code_B = c_Enc(XB), syn_sty_Enc(XB)

            XAA = G(c_code_A, s_code_A)
            XBB = G(c_code_B, s_code_B)

            XBA = G(c_code_B, s_code_A)
            XAB = G(c_code_A, s_code_B)

            c_code_BA, s_code_BA = c_Enc(XBA), real_sty_Enc(XBA)
            c_code_AB, s_code_AB = c_Enc(XAB), syn_sty_Enc(XAB)
            XABA = G(c_code_AB, s_code_A) if lambda_cyc > 0 else 0
            XBAB = G(c_code_BA, s_code_B) if lambda_cyc > 0 else 0

            en_s_code_A = T(s_code_A)
            en_s_code_B = T(s_code_B)
            en_A = G(c_code_B, en_s_code_A)
            en_B = G(c_code_B, en_s_code_B)

            # ── Discriminator ─────────────────────────────────────────────────
            optimizer_D1.zero_grad()
            loss_D1 = (D.compute_loss(XA, valid)
                       + D.compute_loss(XBA.detach(), fake)
                       + D.compute_loss(XB, valid)
                       + D.compute_loss(XAB.detach(), fake))
            loss_D1.backward()
            optimizer_D1.step()

            # ── Generator ─────────────────────────────────────────────────────
            optimizer_G.zero_grad()

            loss_GAN_1          = lambda_gan * D.compute_loss(XBA, valid) + D.compute_loss(XAB, valid)
            loss_ID_1           = lambda_id  * criterion_recon(XAA, XA)
            loss_ID_2           = lambda_id  * criterion_recon(XBB, XB)
            loss_cyc_1          = lambda_cyc * criterion_recon(XABA, XA)
            loss_cyc_2          = lambda_cyc * criterion_recon(XBAB, XB)
            loss_enhanced       = lambda_enhanced * (criterion_recon(en_A, labelB)
                                                     + criterion_recon(en_B, labelB))
            loss_ssim           = lambda_ssim * ((1 - ssim_loss(en_A, labelB))
                                                 + (1 - ssim_loss(en_B, labelB)))
            loss_perceptual     = lambda_perceptual * (perceptual_loss(en_A, labelB)
                                                       + perceptual_loss(en_B, labelB))
            loss_enhanced_latent = lambda_enhanced_latent * criterion_recon(en_s_code_B, en_s_code_A)
            loss_tv             = lambda_tv * (tv_loss(en_B) + tv_loss(en_A))

            loss_G = (loss_GAN_1 + loss_ID_1 + loss_ID_2 + loss_cyc_1 + loss_cyc_2
                      + loss_enhanced + loss_ssim + loss_enhanced_latent
                      + loss_perceptual + loss_tv)
            loss_G.backward()
            optimizer_G.step()

            # ── Logging ───────────────────────────────────────────────────────
            batches_done  = epoch * len(dataloader) + i
            batches_left  = opt.n_epochs * len(dataloader) - batches_done
            time_left     = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % 1000 == 0:
                E = (loss_enhanced.item() + loss_ssim.item()
                     + loss_enhanced_latent.item() + loss_perceptual.item()
                     + loss_tv.item())
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                    " [G loss: %f -- {loss_GAN: %f, loss_Identity: %f,"
                    " Cycle Consistent: %f}]"
                    " [Enhanced loss %f: [L1: %f, ssim: %f, latent: %f,"
                    " perceptual: %f, tv: %f]] ETA: %s"
                    % (epoch, opt.n_epochs, i, len(dataloader),
                       loss_D1.item(), loss_G.item(), loss_GAN_1.item(),
                       loss_ID_1.item() + loss_ID_2.item(),
                       loss_cyc_1.item() + loss_cyc_2.item(),
                       E, loss_enhanced.item(), loss_ssim.item(),
                       loss_enhanced_latent.item(), loss_perceptual.item(),
                       loss_tv.item(), time_left))

        if epoch % opt.sample_interval == 0:
            sample_images(epoch)
            print("Snapshot %d" % epoch)

        lr_scheduler_G.step()
        lr_scheduler_D1.step()

        if epoch % opt.checkpoint_interval == 0 or epoch >= 25:
            torch.save(c_Enc.state_dict(),
                       os.path.join(models_dir, "c_Enc_%d.pth" % epoch))
            torch.save(G.state_dict(),
                       os.path.join(models_dir, "G_%d.pth" % epoch))
            torch.save(real_sty_Enc.state_dict(),
                       os.path.join(models_dir, "real_sty_Enc_%d.pth" % epoch))
            torch.save(syn_sty_Enc.state_dict(),
                       os.path.join(models_dir, "syn_sty_Enc_%d.pth" % epoch))
            torch.save(T.state_dict(),
                       os.path.join(models_dir, "T_%d.pth" % epoch))
            torch.save(D.state_dict(),
                       os.path.join(models_dir, "D_%d.pth" % epoch))


if __name__ == '__main__':
    train()