import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from scipy.io import savemat  # 添加这行导入

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

print('log_dir :', log_dir)
print('model_name:', model_name)

# Flag to control whether to run only FFA or all FFA-FFA4 networks
RUN_ALL_ABLATIONS = False  # Set to True to run all ablation experiments
# Flag to control whether to use multiple loss terms
USE_MULTI_LOSS = True  # Set to True to use PSNR and SSIM in loss calculation

if RUN_ALL_ABLATIONS:
    models_ = {
        'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
        'ffa1': FFA1(gps=opt.gps, blocks=opt.blocks),  # No channel attention, only spatial
        'ffa2': FFA2(gps=opt.gps, blocks=opt.blocks),  # Single channel attention, no spatial
        'ffa3': FFA3(gps=opt.gps, blocks=opt.blocks),  # No attention
        'ffa4': FFA4(gps=opt.gps, blocks=opt.blocks)  # Dual channel attention, no spatial
    }
else:
    models_ = {
        'ffa4': FFA4(gps=opt.gps, blocks=opt.blocks),
    }

loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
    ## 'ots_train':OTS_train_loader,
    ## 'ots_test':OTS_test_loader
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion, model_name):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device);
        y = y.to(opt.device)
        out = net(x)

        # Modified loss calculation with multi-loss flag
        loss = criterion[0](out, y)
        if USE_MULTI_LOSS:
            # Add PSNR and SSIM to loss if flag is True
            with torch.no_grad():  # Temporarily disable gradients for PSNR/SSIM calculation
                psnr_loss = 1.0 / (psnr(out, y) + 1e-8)  # Inverse PSNR as we want to maximize PSNR
                ssim_loss = 1.0 - ssim(out, y)  # Since SSIM is in [0,1], we subtract from 1 to minimize
            loss = loss + 0.1 * psnr_loss + 0.1 * ssim_loss  # Add weighted terms

        if opt.perloss:
            loss2 = criterion[1](out, y)
            loss = loss + 0.04 * loss2

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(
            f'\r{model_name} train loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step, model_name)

            print(f'\n{model_name} step :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, f'./weights/{model_name}_best.pth')
                print(f'\n {model_name} model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    # 修改这里：将npy保存改为mat保存
    savemat(f'./numpy_files/{model_name}_{opt.steps}_losses.mat', {'losses': np.array(losses)})
    savemat(f'./numpy_files/{model_name}_{opt.steps}_ssims.mat', {'ssims': np.array(ssims)})
    savemat(f'./numpy_files/{model_name}_{opt.steps}_psnrs.mat', {'psnrs': np.array(psnrs)})


def test(net, loader_test, max_psnr, max_ssim, step, model_name):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device);
        targets = targets.to(opt.device)
        pred = net(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]

    if RUN_ALL_ABLATIONS:
        # Run all ablation experiments
        for model_key in models_:
            print(f"\n{'=' * 50}")
            print(f"Training {model_key} model...")
            print(f"{'=' * 50}\n")

            net = models_[model_key]
            net = net.to(opt.device)
            if opt.device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            criterion = []
            criterion.append(nn.L1Loss().to(opt.device))
            if opt.perloss:
                vgg_model = vgg16(pretrained=True).features[:16]
                vgg_model = vgg_model.to(opt.device)
                for param in vgg_model.parameters():
                    param.requires_grad = False
                criterion.append(PerLoss(vgg_model).to(opt.device))

            optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),
                                   lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
            optimizer.zero_grad()

            # Create directories if they don't exist
            os.makedirs('./weights', exist_ok=True)
            os.makedirs('./numpy_files', exist_ok=True)

            train(net, loader_train, loader_test, optimizer, criterion, model_key)
    else:
        # Run only the original FFA model
        net = models_[opt.net]
        net = net.to(opt.device)
        if opt.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = []
        criterion.append(nn.L1Loss().to(opt.device))
        if opt.perloss:
            vgg_model = vgg16(pretrained=True).features[:16]
            vgg_model = vgg_model.to(opt.device)
            for param in vgg_model.parameters():
                param.requires_grad = False
            criterion.append(PerLoss(vgg_model).to(opt.device))

        optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),
                               lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
        optimizer.zero_grad()

        # Create directories if they don't exist
        os.makedirs('./weights', exist_ok=True)
        os.makedirs('./numpy_files', exist_ok=True)

        train(net, loader_train, loader_test, optimizer, criterion, opt.net)