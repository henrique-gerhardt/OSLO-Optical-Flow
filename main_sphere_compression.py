import os, sys, inspect
current_frame = inspect.currentframe()
if current_frame is not None:
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(current_frame)))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import dataset
import utils.common_function as common_utils
import utils.pytorch as common_pytorch
import utils.healpix as hp_utils
import utils.weight_transfer as wt_utils
import healpy as hp
import time
from datetime import datetime
from PIL import Image

import shutil
from collections.abc import Iterable
import spherical_models

import healpix_graph_loader
import healpix_sdpa_struct_loader

import warnings


def bgr2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.5870, 0.1140, 0.2989]))

def generateRandomPatches(patch_level_res, n):
    if n == 1:
        n_patches = hp.nside2npix(hp.order2nside(patch_level_res))
        return list(random.sample(range(n_patches), n))

    n_northcap_samples = n_equitorial_samples = n_southcap_samples = n // 3  # We have 3 regions: north cap, equitorial, south cap
    remainder = n % 3

    if remainder:
        n_equitorial_samples += 1
        remainder -= 1

    if remainder:  # if there is still remainder. Now remainder is equal to one
        assert remainder == 1, "the remainder must be equal to one"
        # Randomly sample from north cap or south cap
        val = random.randint(0, 1)
        n_northcap_samples += val
        n_southcap_samples += 1 - val

    list_equitorial = random.sample(list(hp_utils.get_regionPixelIds(patch_level_res, "equatorial_region", nest=True)), n_equitorial_samples)
    list_north_cap  = random.sample(list(hp_utils.get_regionPixelIds(patch_level_res, "north_polar_cap", nest=True)), n_northcap_samples)
    list_south_cap  = random.sample(list(hp_utils.get_regionPixelIds(patch_level_res, "south_polar_cap", nest=True)), n_southcap_samples)

    list_patch_ids = []
    list_patch_ids.extend(list_equitorial)
    list_patch_ids.extend(list_north_cap)
    list_patch_ids.extend(list_south_cap)
    return list_patch_ids


class RateDistortionLoss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, num_nodes, _ = target.size()
        out = {}
        num_pixels = N * num_nodes

        # Total number of bits divided by number of pixels
        out['bpp'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'].values())
        # Mean squared error across pixels.
        out['mse'] = self.mse(output['x_hat'], target)
        # Multiply by 255^2 to correct for rescaling.
        out['mse'] *= 255 ** 2

        # The rate-distortion cost.
        out['loss'] = self.lmbda * out['mse'] + out['bpp']

        return out


def train_epoch(train_dataloader, struct_loader, model, criterion, optimizer, optimizer_aux, patch_res, n_patch_per_sample, print_freq, epoch, clip_max_norm, folder_plot_grad, single_conv:bool=False, ar_hops:int=2):
    model.train()
    device = next(model.parameters()).device
    batch_time = common_utils.AverageMeter('Batch processing time', ':6.3f')
    data_time = common_utils.AverageMeter('Data Loading time', ':6.3f')
    loss = common_utils.AverageMeter('Loss', ':.4e')
    bpp = common_utils.AverageMeter('BPP', ':.4e')
    mse = common_utils.AverageMeter('MSE', ':.4e')
    aux_loss = common_utils.AverageMeter('Loss AUX', ':.4e')

    sample_res = train_dataloader.dataset.resolution
    n_patches, nPix_per_patch = struct_loader.getPatchesInfo(sampling_res=sample_res, patch_res=patch_res)
    noPatching = True if n_patches == 1 else False
    healpix_resolution_patch_level = hp.nside2order(hp.npix2nside(n_patches)) if not noPatching else None
    list_res = [sample_res + offset if noPatching else (sample_res + offset, patch_res + offset) for offset in model.get_resOffset()]

    progress = common_utils.ProgressMeter(len(train_dataloader), [batch_time, data_time, loss, aux_loss], prefix=f"Epoch: [{epoch+1}]")

    end = time.time()
    
    for batch_id, input in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        list_patch_ids = generateRandomPatches(healpix_resolution_patch_level, n_patch_per_sample) if not noPatching else [0]
        # list_patch_ids = [170]
        # print(list_patch_ids)

        for patch_id in list_patch_ids:
            optimizer.zero_grad()
            optimizer_aux.zero_grad()

            dict_index = dict()
            dict_weight = dict()
            for r in list_res:
                if struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                    hops = max(2 if single_conv else 1, ar_hops)
                    dict_index[r], dict_weight[r], _, _, _ = struct_loader.getStruct(sampling_res=r if noPatching else r[0], num_hops=hops, patch_res=None if noPatching else r[1], patch_id=patch_id)
                else:
                    dict_index[r], dict_weight[r], _, _ = struct_loader.getGraph(sampling_res=r if noPatching else r[0], patch_res=None if noPatching else r[1], num_hops=0, patch_id=patch_id)

            if noPatching:
                d = input["features"].to(device)
            else:
                d = input["features"].narrow(dim=1, start=patch_id * nPix_per_patch, length=nPix_per_patch).to(device)

            out_net = model(d, dict_index, dict_weight, sample_res, patch_res)
            # print("out_net['x_hat'].max()=", out_net['x_hat'].max(), "out_net['x_hat'].min()=", out_net['x_hat'].min())
            # print("d.max()=", d.max(), "d.min()=", d.min())

            out_criterion = criterion(out_net, d)
            loss_aux = model.aux_loss()
            out_criterion['loss'].backward()
            if clip_max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            loss_aux.backward()

            if folder_plot_grad:
                fileaddr = os.path.join(folder_plot_grad, f"epoch_{epoch:03d}_batchID_{batch_id:03d}")
                common_pytorch.plot_grad_flow(model.named_parameters(), fileaddr)

            optimizer.step()
            optimizer_aux.step()

            loss.update(out_criterion['loss'].detach().item())
            mse.update(out_criterion['mse'].detach().item())
            bpp.update(out_criterion['bpp'].detach().item())
            aux_loss.update(loss_aux.detach().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id+1) % print_freq == 0:
            progress.display(batch_id+1)

    return {"loss": loss.avg, "aux_loss": aux_loss.avg, "bpp": bpp.avg, "mse":mse.avg}

def compute_theoretical_bits(out_net):
    list_latent_bits = [torch.ceil((torch.log(likelihoods).sum(dim=(1, 2)) / (-math.log(2)))) for likelihoods in out_net['likelihoods'].values()]
    total_bits_per_image = torch.sum(torch.stack(list_latent_bits, dim=0), dim=0).detach().cpu().long()
    return total_bits_per_image

def compute_actual_bits(compressed_stream):
    list_latent_bits = [torch.tensor([len(s) * 8 for s in list_s]) for list_s in compressed_stream["strings"]]
    total_bits_per_image = torch.sum(torch.stack(list_latent_bits, dim=0), dim=0).detach().cpu().long()
    return total_bits_per_image

def test_epoch(test_dataloader, struct_loader, model, criterion, patch_res=None, visFolder=None, print_freq=None, epoch=None, checkWithActualCompression=False, only_npy=False, single_conv:bool=False, ar_hops:int=2, save_color:bool=False, skip_tested:bool=False):
    model.eval()
    device = next(model.parameters()).device

    batch_time = common_utils.AverageMeter('Batch processing time', ':6.3f')
    data_time = common_utils.AverageMeter('Data Loading time', ':6.3f')
    loss = common_utils.AverageMeter('Loss', ':.4e')
    bpp = common_utils.AverageMeter('BPP', ':.4e')
    mse = common_utils.AverageMeter('MSE', ':.4e')
    aux_loss = common_utils.AverageMeter('Loss AUX', ':.4e')

    if (patch_res is not None) and (patch_res < 0):
        patch_res = None
    sample_res = test_dataloader.dataset.resolution
    n_patches, nPix_per_patch = struct_loader.getPatchesInfo(sampling_res=sample_res, patch_res=patch_res)
    noPatching = True if n_patches == 1 else False
    list_res = [sample_res + offset if noPatching else (sample_res + offset, patch_res + offset) for offset in model.get_resOffset()]

    if print_freq is not None:
        progress = common_utils.ProgressMeter(len(test_dataloader), [batch_time, data_time, loss, aux_loss, mse],
                                              prefix=f"Test/Validation Epoch: [{epoch if epoch is None else epoch+1}]")

    tested_imgs = []
    if visFolder:
        os.makedirs(visFolder, exist_ok=True)
        if skip_tested:
            tested_imgs_rates = []
            rates_file = os.path.join(visFolder, "rates.txt")
            if os.path.exists(rates_file):
                print(rates_file)
                with open(rates_file) as f:
                    for line in f:
                        if line.startswith("#"): continue
                        tested_imgs_rates.append(line.split()[0])
            
            for img in tested_imgs_rates:
                if os.path.exists(os.path.join(visFolder, img + (f"_epoch_{epoch:03d}" if epoch is not None else "") + "_reconstructed.npy")):
                    tested_imgs.append(img)
            if tested_imgs: print(f"Skipping {len(tested_imgs)} already tested images")
        
        if checkWithActualCompression:
            if not tested_imgs:
                with open(os.path.join(visFolder, "rates.txt"), 'w') as f:
                    f.write("#This line is a comment. The first column is the filename, the second is the estimated rate, and the third is the actual rate\n")

    end = time.time()
    with torch.no_grad():   # even though we are in eval mode this torch.no_grad() will additionally save some memory.
        for batch_id, input in enumerate(test_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            test_ids = list(range(len(input["features"]))) # images inside batch to test
            if skip_tested:
                test_ids = [i for i in test_ids if common_utils.extract_filename(input["filename"][i]) not in tested_imgs]
            
            if visFolder:
                reconstructed_output = torch.empty((len(test_ids), *input["features"].size()[1:]), dtype=input["features"].dtype, device=torch.device("cpu"))

            if checkWithActualCompression:
                total_bits_theoretical = torch.zeros(len(test_ids), dtype=torch.long, device=torch.device("cpu"))
                total_bits_actual = torch.zeros(len(test_ids), dtype=torch.long, device=torch.device("cpu"))

            if test_ids:
                for patch_id in range(n_patches):
                    dict_index = dict()
                    dict_weight = dict()
                    for r in list_res:
                        if struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            hops = max(2 if single_conv else 1, ar_hops)
                            if noPatching:
                                dict_index[r], dict_weight[r], _ = struct_loader.getStruct(sampling_res=r, num_hops=hops, patch_res=None, patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = struct_loader.getStruct(sampling_res=r[0], num_hops=hops, patch_res=r[1], patch_id=patch_id)
                        else:
                            if noPatching:
                                dict_index[r], dict_weight[r] = struct_loader.getGraph(sampling_res=r, patch_res=None, num_hops=0, patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _ = struct_loader.getGraph(sampling_res=r[0], patch_res=r[1], num_hops=0, patch_id=patch_id)

                    if noPatching:
                        d = input["features"][test_ids,:].to(device)
                    else:
                        d = input["features"][test_ids,:].narrow(dim=1, start=patch_id * nPix_per_patch, length=nPix_per_patch).to(device)

                    out_net = model(d, dict_index, dict_weight, sample_res, patch_res)
                    out_criterion = criterion(out_net, d)
                    loss_aux = model.aux_loss()
                    
                    if checkWithActualCompression:
                        out_net['x_hat'].clamp_(0, 1)
                        theoretical_rates = compute_theoretical_bits(out_net)
                        total_bits_theoretical += theoretical_rates

                        # output of real compression and decompression
                        compressed = model.compress(d, dict_index, dict_weight, sample_res, patch_res)
                        actual_rates = compute_actual_bits(compressed)
                        total_bits_actual += actual_rates
                        decompressed = model.decompress(compressed["strings"], compressed[ "shape"], dict_index, dict_weight, sample_res, patch_res)
                        decompressed['x_hat'].clamp_(0, 1)

                        diff = (out_net["x_hat"] - decompressed["x_hat"]).abs()
                        diff_in_bits = (theoretical_rates - actual_rates).abs()
                        print(f"Patch [{patch_id+1:03d}/{n_patches:03d}]")
                        print(f"max difference={diff.max()}, min difference={diff.min()}")
                        print(f"diff in bits={diff_in_bits}, ratio (compressed/training)={torch.div(actual_rates, theoretical_rates)}%", flush=True)
                        
                        isCloseReconstruction = torch.allclose(out_net["x_hat"], decompressed["x_hat"], atol=1e-06, rtol=0)
                        if not isCloseReconstruction:
                            warnings.warn("The output of decompressed image is not equal to image")
                        isCloseBits = torch.allclose(theoretical_rates, actual_rates, atol=0, rtol=1e-2)
                        if not isCloseBits:
                            warnings.warn("The number of compressed bits is not equal to the number of bits computed in training phase")

                    aux_loss.update(loss_aux)
                    bpp.update(out_criterion['bpp'])
                    loss.update(out_criterion['loss'])
                    mse.update(out_criterion['mse'])

                    if visFolder:
                        if noPatching:
                            reconstructed_output.copy_(out_net['x_hat'].detach().cpu())
                        else:
                            reconstructed_output.narrow(dim=1, start=patch_id * nPix_per_patch, length=nPix_per_patch).copy_(out_net['x_hat'].detach().cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (print_freq is not None) and ((batch_id+1) % print_freq == 0):
                progress.display(batch_id+1)

            if visFolder:
                if checkWithActualCompression:
                    with open(os.path.join(visFolder, "rates.txt"), 'a') as f:
                        for i, img_id in enumerate(test_ids):
                            img_name, bits_theoretical, bits_actual = common_utils.extract_filename(input["filename"][img_id]), total_bits_theoretical[i].item(), total_bits_actual[i].item()
                            f.write(f"{img_name}\t{bits_theoretical}\t{bits_actual}\n")
                
                # d_test = input["features"].to(device)
                # out_net_test = model(d_test)
                for i, img_id in enumerate(test_ids):
                    filename_common = common_utils.extract_filename(input["filename"][img_id])
                    if not only_npy:
                        fileAddress_original = os.path.join(visFolder, filename_common + "_original.png")
                        if not os.path.isfile(fileAddress_original):  # if original file does not exist
                            colors_original = input["features"][img_id].detach().cpu().numpy()
                            hp.visufunc.mollview(bgr2gray(colors_original), fig=1, cmap=plt.cm.gray, nest=True, min=0., max=1., title="Original gray", xsize=1200, cbar=False) # type: ignore
                            plt.savefig(fileAddress_original, bbox_inches='tight', dpi=300)
                            plt.close()

                    colors_reconstructed = reconstructed_output[i].detach().cpu().numpy()
                    filename_reconstructed = filename_common + (f"_epoch_{epoch:03d}" if epoch is not None else "") + "_reconstructed"
                    if not only_npy:
                        # Save in png mollview projection
                        hp.visufunc.mollview(bgr2gray(colors_reconstructed), fig=2, cmap=plt.cm.gray, nest=True, min=0., max=1., title="Reconstructed gray", xsize=1200, cbar=False) # type: ignore
                        plt.savefig(os.path.join(visFolder, filename_reconstructed + ".png"), bbox_inches='tight', dpi=300)
                    # Save in numpy array
                    colors_rec_np = np.rint(colors_reconstructed * 255.).astype(np.uint8)
                    if save_color:
                        img = hp_utils.mollw(colors_rec_np, 2276, bilinear_interpolation=True, isNest=True, bgcolor='white')
                        img = Image.fromarray(img, 'RGB')
                        img.save(os.path.join(visFolder, filename_reconstructed + "_color" + ".png"))
                    np.save(os.path.join(visFolder, filename_reconstructed + ".npy"), colors_rec_np)
                    plt.close()

    return {"loss": loss.avg, "aux_loss": aux_loss.avg, "bpp": bpp.avg, "mse": mse.avg}

def save_config(list_dicts, output_folder, filename=None):
    os.makedirs(output_folder, exist_ok=True)

    output_fileAddr = os.path.join(output_folder, filename) if filename else os.path.join(output_folder, "config.txt")

    with open(output_fileAddr, 'w') as f:
        for dict in list_dicts:
            for key,val in dict.items():
                f.write(f"{key:_<40}: {str(val)}\n")  # check this for all kinds of formatting
            f.write("="*60+"\n")

def save_checkpoint(state_dics, is_best, output_folder, filename=None, only_best_model=False, best_checkpoint_milestones=None):
    os.makedirs(output_folder, exist_ok=True)
    if not only_best_model:
        if filename:
            if filename.endswith('.tar'): filename = filename.replace('.tar', '')
            if filename.endswith('.pth'): filename = filename.replace('.pth', '')
        else:
            filename = f"checkpoint_{state_dics['epoch']+1:04d}"
        
        output_fileAddr = os.path.join(output_folder, filename+'.pth.tar')
        torch.save(state_dics, output_fileAddr)
        print(f'saved checkpoint to {output_fileAddr}')
    if is_best:
        fp_best = os.path.join(output_folder, 'checkpoint_best_loss.pth.tar')
        if only_best_model:
            torch.save(state_dics, fp_best)
        else:
            shutil.copyfile(output_fileAddr, fp_best)
        print(f'saved best model to {fp_best}')
        if best_checkpoint_milestones:
            for milestone in best_checkpoint_milestones:
                if state_dics["epoch"] < milestone:
                    fp_milestone = os.path.join(output_folder, f'checkpoint_best_loss_{milestone:03d}.pth.tar')
                    shutil.copyfile(fp_best, fp_milestone)
                    print(f'saved best model to {fp_milestone}')
                    break

def get_model_name(string_name):
    for ch in [",", " ", "_", "-"]: # replace all with space
        string_name=string_name.replace(ch, ' ')
    list_of_words = string_name.split()
    result = list_of_words[0][0].upper()+list_of_words[0][1:]   # capitalize the first letter of the first word no matter the rest
    for word in list_of_words[1:]:
        result = result + word[0].upper() + word[1:] # capitalize the first letter of word no matter the rest
    return result

quality_cfgs = {
    **dict.fromkeys(['SphereFactorizedPrior', 'SphereScaleHyperprior', 'SphereMeanScaleHyperprior'],{    # all models have same parameters
                                                                        1: (128, 192),
                                                                        2: (128, 192),
                                                                        3: (128, 192),
                                                                        4: (128, 192),
                                                                        5: (128, 192),
                                                                        6: (142, 270),
                                                                        7: (142, 270),
                                                                        8: (142, 270),
                                                                        }),
}

parser = argparse.ArgumentParser()
# Verbosity
parser.add_argument('--foldername-valtest', type=str, help='Local folder to save validation/test set')
parser.add_argument('--foldername-plot-gradient', type=str, help='Local folder to save plots of gradients')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--interval-save-valtest', type=int, default=20, help='Interval to be used for storing data in mollview')
parser.add_argument('--only-npy-valtest', action='store_true', help='Only store .npy files in foldername-valtest during test/validation phase')
parser.add_argument('--valtest-color', action='store_true', help='Store reconstructed color images additionally during test/validation phase')
parser.add_argument('--skip-tested', '-st', action='store_true', help="Skip already tested images in case of interrupted testing. Default is False (Full testing procedure).")
# Saving config
parser.add_argument('--checkpoint-interval', type=int, default=10, help='Interval to be used for saving data either for inference or resuming training')
parser.add_argument('--checkpoint-file', type=str, help='File address to resume training from the previous saved checkpoint')
parser.add_argument('--loss-file', type=str, default='loss.txt', help='File name to store loss values after every epoch. Empty string for no saving.')
parser.add_argument('--filename-test-results', type=str, default='test_results', help='File name to store test loss values. Will be stored as <filename_test_results>.npz')
parser.add_argument("--out-dir", "-o", type=str, default="./", help="Output directory to save model and results.")
parser.add_argument("--neighbor-struct-dir", "-nd", type=str, default="../GraphData", help="Directory to save/load neighboring structures.")
parser.add_argument('--seed', type=int, help='Set random seed for reproducibility')
# Dataset
parser.add_argument('--train-data', '-i', type=str, default='./train.txt', help="A text file containing location of train images.")
parser.add_argument('--test-data', '-t', type=str, help="A text file containing location of test images. This represents test mode (no training will be done)")
parser.add_argument('--validation-data', '-v', type=str, help="A text file containing location of validation images.")
parser.add_argument('--tmp-dir', '-tmp', type=str, default='', help="A temporary path where images are saved.")
parser.add_argument('--batch-size-train', '-bst', type=int, default=10, help='batch size train dataset')
parser.add_argument('--batch-size-valtest', '-bsvt', type=int, default=10, help='batch size validation and test datasets')
parser.add_argument('--dataloader-num-workers', type=int, default=0, help='multi-process data loading with the specified number of loader worker processes')
parser.add_argument("--healpix-res", "-hr", type=int, default=10, help="Resolution of the healpix for sampling.")
parser.add_argument("--patch-res-train", "-prt", type=int, default=8, help="Resolution of the healpix patches (Negative value means no patching). For example, a value equal to 8 means patches of size 2^(8) x 2^(8) = 256x256 (as explained in Balle's paper.")
parser.add_argument("--patch-res-valtest", "-prvt", type=int, default=8, help="Resolution of the healpix patches (Negative value means no patching). For example, a value equal to 8 means patches of size 2^(8) x 2^(8) = 256x256 (as explained in Balle's paper.")
parser.add_argument("--n-patch-per-sample", "-np", type=int, default=1, help="Number of patches taken from each sample each time.")
# Architecture config
parser.add_argument('--model', '-m', default='SphereScaleHyperprior', type=str, choices=['SphereFactorizedPrior','SphereScaleHyperprior','SphereMeanScaleHyperprior'], help='Model name to choose')
parser.add_argument('--attention', '-at', action='store_true', help='use additional attention modules in the En- and Decoder')
parser.add_argument('--nonlinearity', '-nl', default='GDN', type=str, choices=['GDN', 'RB'], help='Nonlinearity used in the En- and Decoder between convoutions. GDN: Generalized Divisive Normalization, RB: 3 Residual Blocks')
parser.add_argument('--context-model', '-cm', default='', type=str, choices=['', 'autoregressive', 'full'], help="Context model to use. Empty string is no context model. 'full' is not causal (no masked convolution) and thus just for reference.")
parser.add_argument('--autoregressive-hops', '-arhops', type=int, default=1, help='Number of hops for neighborhood for autoregressive convolution in context model. Default is 1 (3x3 2D equivalent).')
parser.add_argument('--quality', '-q', default=4, type=int, choices=range(1, 9), help='Quality levels (1: lowest, highest: 8)')
parser.add_argument('--lambda', dest='lmbda', type=float, default=1e-2, help='Bit-rate distortion parameter (default: %(default)s)')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate (default: %(default)s)')
parser.add_argument('--learning-rate-aux', '-lraux', type=float, default=1e-3, help='Auxiliary loss learning rate (default: %(default)s)')
parser.add_argument('--clip_max_norm', type=float, default=1., help='gradient clipping max norm')
# Network config
parser.add_argument('--max-epochs', '-e', type=int, default=1000, help='max epochs')
parser.add_argument('--no-scheduler', '-ns', action='store_true', help='Disable scheduler')
parser.add_argument('--scheduler-milestones', nargs='+', type=int, default=[30, 130], help=" List of epoch indices for scheduler to adjust learning rate if no validation data is given.")
parser.add_argument('--scheduler-steplr', type=int, default=-1, help="Use StepLR scheduler if set positive step size is set.")
parser.add_argument('--scheduler-gamma', type=float, default=np.sqrt(0.1), help='Multiplicative factor of learning rate decay')
parser.add_argument('--scheduler-patience', type=int, default=10, help='Patience in terms of number of epochs for scheduler ReduceLROnPlateau')
parser.add_argument('--validation-start', type=int, default=1, help='first epoch using validation for scheduler ReduceLROnPlateau. Allows to start validation after some epochs. Default is 1 (validation from beginning).')
parser.add_argument('--gpu', '-g', action='store_true', help='enables cuda')
parser.add_argument('--gpu_id', '-gid', type=int, default=-1, help='select cuda device by index. Default is -1 (cuda).')
parser.add_argument('--conv', '-c',  type=str, default='SDPAConv', help="Graph convolution method")
parser.add_argument('--skip-connection-aggregation', '-sc',  type=str, default='sum', help="Mode for jumping knowledge")
parser.add_argument('--single-conv',  action='store_true', help="Use only one convolution for SDPAConv")
parser.add_argument('--pool-func', '-pf', type=str, default="stride", help="Pooling function.")
parser.add_argument('--unpool-func', '-upf', type=str, default="pixel_shuffle", help="Unpooling function. One of ['pixel_shuffle' (Default, baseline implementation), 'SDPAConvTransposed' (recommended, lower #params), 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']")
# HealPix
parser.add_argument("--use-4connectivity", action='store_true', help='use 4 neighboring for graph construction')
parser.add_argument("--use-euclidean", action='store_true', help='Use geodesic distance for graph weights')
parser.add_argument("--weight-type", '-w', type=str, default='identity', help="Weighting function on distances between nodes of the graph")
# SDPA
parser.add_argument("--sdpa-normalization", '-sn', type=str, default='non', help="normalization method for sdpa convolutions")
# Weight Transfer
parser.add_argument("--weight-transfer", '-wt', action='store_true', help='Transfer weights from model pretrained on plain images')
parser.add_argument("--foldername-pretrained", type=str, default='./pretrained', help='Local folder to save pretrained model')
parser.add_argument('--save-best-milestones', nargs='*', type=int, default=[100, 200, 400], help="List of epoch indices to store best checkpoints separately (only for weight transfer).")

def main():
    print('='*40+f' {datetime.now().strftime("%d-%m-%Y %H:%M:%S")} '+'='*40)
    args = parser.parse_args()
    if not args.weight_transfer:
        args.save_best_milestones = None
    if args.save_best_milestones:
        args.save_best_milestones.sort()

    print("=========== printing args ===========")
    for key, val in args.__dict__.items():
        print(f"{key:_<40}: {val}\n")  # check this for all kinds of formatting
    print("=" * 60 + "\n")

    seedValue = random.randrange(2**32)   # create a seed
    if args.seed is not None:
        seedValue = args.seed
    elif args.checkpoint_file and os.path.isfile(os.path.join(args.out_dir, 'config.txt')):
        with open(os.path.join(args.out_dir, 'config.txt'), 'r') as f:
            config_lines = f.readlines()
            for line in config_lines:
                if line.strip().startswith('seedValue'):
                    seedValue = int(line.split(': ')[1])
    common_pytorch.set_seed(seedValue)
    
    device_str = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device_str = 'cuda'
        if args.gpu_id >= 0:
            device_str = f'cuda:{args.gpu_id}'
    device = torch.device(device_str)
    print("Data will be processed on", device)

    # Healpix related parameters
    if args.conv == "SDPAConv":
        struct_loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type=args.weight_type, use_geodesic=not args.use_euclidean, use_4connectivity=args.use_4connectivity, normalization_method=args.sdpa_normalization, cutGraphForPatchOutside=True, load_save_folder=args.neighbor_struct_dir)
    else:
        struct_loader = healpix_graph_loader.HealpixGraphLoader(weight_type=args.weight_type, use_geodesic=not args.use_euclidean, use_4connectivity=args.use_4connectivity, load_save_folder=args.neighbor_struct_dir)

    model_name = get_model_name(args.model)
    print("model=", model_name)
    model = getattr(spherical_models, model_name)
    if args.weight_transfer: # q>=6: 192, 320
        from compressai.zoo.image import cfgs
        # model_name SphereFactorizedPrior -> model_key factorizedPrior
        model_key = model_name.split('Sphere')[-1]
        model_key = model_key[0].lower() + model_key[1:]
        N, M = cfgs[wt_utils.MODELS_PRETRAINED[model_key]][args.quality]
    else:
        N, M = quality_cfgs[model_name][args.quality]

    model_args = [
        M,
        args.conv,
        args.skip_connection_aggregation,
        args.pool_func,
        args.unpool_func,
        args.attention,
        args.nonlinearity,
        args.single_conv,
        args.context_model,
        args.autoregressive_hops,
    ]
    if args.skip_connection_aggregation == "cat": # To have almost the same number of parameters as sum or max aggregation
        net = model(2*N, *model_args).to(device)
    else:
        net = model(N, *model_args).to(device)

    # Use list of tuples instead of dict to be able to later check the elements are unique and there is no intersection
    parameters = [(n, p) for n, p in net.named_parameters() if not n.endswith(".quantiles")]
    aux_parameters = [(n, p) for n, p in net.named_parameters() if n.endswith(".quantiles")]

    # Make sure we don't have an intersection of parameters
    parameters_name_set = set(n for n, _ in parameters)
    aux_parameters_name_set = set(n for n, _ in aux_parameters)
    assert len(parameters) == len(parameters_name_set)
    assert len(aux_parameters) == len(aux_parameters_name_set)

    inter_params = parameters_name_set & aux_parameters_name_set
    union_params = parameters_name_set | aux_parameters_name_set

    assert len(inter_params) == 0
    assert len(union_params) - len(dict(net.named_parameters()).keys()) == 0

    optimizer = torch.optim.Adam((p for (_, p) in parameters if p.requires_grad), lr=args.learning_rate)
    optimizer_aux = torch.optim.Adam((p for (_, p) in aux_parameters if p.requires_grad), lr=args.learning_rate_aux)

    n_parameters_dict = dict()
    n_parameters_dict["# model parameters"] = sum(p.numel() for (_, p) in parameters if p.requires_grad)
    n_parameters_dict["# entropy bottleneck(s) parameters"] = sum(p.numel() for (_, p) in aux_parameters if p.requires_grad)
    for key, val in n_parameters_dict.items():
        print(f'{key}: {common_utils.human_readable_number(val)}')

    # print("=================================")
    # print("parameter names:")
    # for n, p in net.named_parameters():
    #     if (p.requires_grad) and ("bias" not in n):
    #         print(n)
    # print("=================================")

    if not args.no_scheduler and not args.test_data:
        if args.scheduler_steplr > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_steplr, gamma=args.scheduler_gamma)
            scheduler_aux = torch.optim.lr_scheduler.StepLR(optimizer_aux, args.scheduler_steplr, gamma=args.scheduler_gamma)
        elif args.validation_data:  # validation data
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_gamma, patience=args.scheduler_patience, verbose=True, threshold=0.0001)
            scheduler_aux = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_aux, factor=args.scheduler_gamma, patience=args.scheduler_patience, verbose=True, threshold=0.0001)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)
            scheduler_aux = torch.optim.lr_scheduler.MultiStepLR(optimizer_aux, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    list_mean_losses = [None] * args.max_epochs
    list_mean_losses_validation = [None] * args.max_epochs
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = -1
    best_loss = 1e10

    if args.checkpoint_file:  # load from previous checkpoint
        print("Loading", args.checkpoint_file)
        checkpoint = torch.load(args.checkpoint_file, map_location=device, weights_only=False)
        last_epoch = checkpoint["epoch"]
        net.load_state_dict((checkpoint["net_state_dict"]))
        net.update(force=True)  # update the model CDFs parameters.
        net.to(device)
        optimizer.load_state_dict((checkpoint["optimizer_state_dict"]))
        optimizer_aux.load_state_dict((checkpoint["optimizer_aux_state_dict"]))

        list_mean_losses = checkpoint["list_mean_losses"]
        epoch_delta = args.max_epochs - len(list_mean_losses)
         # allow for resuming training with more epochs
        if epoch_delta > 0: list_mean_losses.extend([None] * (epoch_delta))

        if "list_mean_losses_validation" in checkpoint:
            list_mean_losses_validation = checkpoint["list_mean_losses_validation"]
            if epoch_delta > 0: list_mean_losses_validation.extend([None] * (epoch_delta))
        
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
            print("best_loss loaded=", best_loss)

        if not args.no_scheduler and not args.test_data:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict((checkpoint["scheduler_state_dict"]))
                if args.validation_start > 1:
                    assert scheduler.last_epoch == max(0, last_epoch - args.validation_start + 2), "scheduler last epoch should fit to validation_start" # type: ignore
                else:
                    assert scheduler.last_epoch == last_epoch + 1, "scheduler should have same last epoch" # type: ignore
            else:
                scheduler.last_epoch = last_epoch + 1 # type: ignore
            if "scheduler_aux_state_dict" in checkpoint:
                scheduler_aux.load_state_dict((checkpoint["scheduler_aux_state_dict"]))
                if args.validation_start > 1:
                    assert scheduler_aux.last_epoch == max(0, last_epoch - args.validation_start + 2), "scheduler last epoch should fit to validation_start" # type: ignore
                else:
                    assert scheduler_aux.last_epoch == last_epoch + 1, "scheduler should have same last epoch" # type: ignore
            else:
                scheduler_aux.last_epoch = last_epoch + 1 # type: ignore
        print("last_epoch loaded=", last_epoch)
        
    elif args.weight_transfer: # load from pretrained CompressAI model
        fp_pretrained = wt_utils.download_pretrained_model(model_name, args.lmbda, args.quality, args.foldername_pretrained)
        print("Loading", fp_pretrained)
        pretrained = torch.load(fp_pretrained, map_location=device, weights_only=False)
        pretrained = wt_utils.transform_state_dict(pretrained, args.single_conv)
        net.load_state_dict(pretrained)
        net.update(force=True)  # update the model CDFs parameters.
        net.to(device)

    if args.test_data:  # test phase. Note: this should be after args.checkpoint_file to load the latest model
        if not args.checkpoint_file: net.update(force=True)  # update the model CDFs parameters.
        transform = torchvision.transforms.Compose([dataset.ToTensor(to_range_minusOne_to_plusOne=False)])
        test_dataset = dataset.HealpixDataset(args.test_data, transform=transform, tmp_dir=args.tmp_dir)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_valtest, shuffle=False, num_workers=args.dataloader_num_workers)
        assert test_dataset.resolution == args.healpix_res, "resolution of test dataset doesn't match with input dataset"
        valTestVisFolder = os.path.join(args.out_dir, args.foldername_valtest) if args.foldername_valtest is not None else None

        loss_test = test_epoch(test_dataloader, struct_loader, net, criterion, args.patch_res_valtest, valTestVisFolder, args.print_freq, checkWithActualCompression=True, only_npy=args.only_npy_valtest, single_conv=args.single_conv, ar_hops=args.autoregressive_hops, save_color=args.valtest_color, skip_tested=args.skip_tested)
        print("Loss test:", ', '.join([f'{k}={v:6.4f}' for k, v in loss_test.items()]), flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        output_fileAddr = os.path.join(args.out_dir, args.filename_test_results+'.npz')
        for val in loss_test.values(): # save only for nonzero values (zero values imply skipped testing)
            if val != 0:
                np.savez(output_fileAddr, loss_test=loss_test)
        return
    
    transform = torchvision.transforms.Compose([dataset.ToTensor(to_range_minusOne_to_plusOne=False)])
    train_dataset = dataset.HealpixDataset(args.train_data, transform=transform, tmp_dir=args.tmp_dir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.dataloader_num_workers)
    assert train_dataset.resolution == args.healpix_res, "resolution of train dataset doesn't match with input dataset"

    if args.validation_data: # validation data
        transform = torchvision.transforms.Compose([dataset.ToTensor(to_range_minusOne_to_plusOne=False)])
        validation_dataset = dataset.HealpixDataset(args.validation_data, transform=transform)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size_valtest, shuffle=False, num_workers=args.dataloader_num_workers)
        assert validation_dataset.resolution == args.healpix_res, "resolution of validation dataset doesn't match with input dataset"
    # save loss
    if args.loss_file:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, args.loss_file), 'a') as f:
            f.write('='*40+f' {datetime.now().strftime("%d-%m-%Y %H:%M:%S")} '+'='*40+'\n')
    # to visualize gradient status
    plotGradFolder = os.path.join(args.out_dir, args.foldername_plot_gradient) if args.foldername_plot_gradient is not None else None
    if plotGradFolder:
        os.makedirs(plotGradFolder, exist_ok=True)
    perform_validation = False
    for epoch in range(last_epoch + 1, args.max_epochs): # epoch=0...max_epochs-1, printing and checkpoint saving in 1...max_epochs
        loss_train = train_epoch(train_dataloader, struct_loader, net, criterion, optimizer, optimizer_aux, args.patch_res_train, args.n_patch_per_sample, args.print_freq, epoch, args.clip_max_norm, plotGradFolder, args.single_conv, ar_hops=args.autoregressive_hops)

        if epoch == 0:
            print(f"saving config.txt file in {args.out_dir}", flush=True)
            save_config([args.__dict__, n_parameters_dict, {"seedValue":seedValue}], args.out_dir, filename="config.txt")

        list_mean_losses[epoch] = loss_train
        loss_str = f"Loss train: Epoch [{epoch+1:04n}/{args.max_epochs:04n}]: " + ', '.join([f'{k}={v:6.4f}' for k, v in loss_train.items()])
        print(loss_str, flush=True)
        if args.loss_file:
            with open(os.path.join(args.out_dir, args.loss_file), 'a') as f:
                f.write(loss_str + '\n')
        
        perform_validation = args.validation_data and ((epoch + 1) >= args.validation_start)
        if args.validation_data:  # validation data
            if perform_validation:
                saveVis = (((epoch+1) % args.interval_save_valtest == 0) or ((epoch+1) == args.max_epochs)) and (args.foldername_valtest is not None)
                valTestVisFolder = os.path.join(args.out_dir, args.foldername_valtest) if saveVis else None
                loss_validation = test_epoch(validation_dataloader, struct_loader, net, criterion, args.patch_res_valtest, valTestVisFolder, args.print_freq, epoch, only_npy=args.only_npy_valtest, single_conv=args.single_conv, ar_hops=args.autoregressive_hops, save_color=args.valtest_color)
                list_mean_losses_validation[epoch] = loss_validation
                loss_str = f"Loss validation: Epoch [{epoch+1:04n}/{args.max_epochs:04n}]: " + ', '.join([f'{k}={v:6.4f}' for k, v in loss_validation.items()])
                print(loss_str, flush=True)
                if args.loss_file:
                    with open(os.path.join(args.out_dir, args.loss_file), 'a') as f:
                        f.write(loss_str + '\n')
                if not args.no_scheduler:
                    scheduler.step(loss_validation["loss"])
                    scheduler_aux.step(loss_validation["aux_loss"])
        else:
            if not args.no_scheduler:
                scheduler.step() # type: ignore
                scheduler_aux.step() # type: ignore
        
        save_regular_checkpoint = (epoch+1) % args.checkpoint_interval == 0
        if save_regular_checkpoint or perform_validation:
            is_best = loss_validation["loss"] < best_loss if perform_validation else False
            best_loss = min(loss_validation["loss"], best_loss) if perform_validation else best_loss
            states = {
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aux_state_dict': optimizer_aux.state_dict(),
                "list_mean_losses": list_mean_losses,
                "list_mean_losses_validation": list_mean_losses_validation,
                "best_loss": best_loss,
            }
            if not args.no_scheduler:
                states["scheduler_state_dict"] = scheduler.state_dict()
                states["scheduler_aux_state_dict"] = scheduler_aux.state_dict()
            save_checkpoint(states, is_best, args.out_dir, only_best_model=(not save_regular_checkpoint), best_checkpoint_milestones=args.save_best_milestones)

    # Saving last results
    is_best = loss_validation["loss"] < best_loss if perform_validation else False
    best_loss = min(loss_validation["loss"], best_loss) if perform_validation else best_loss
    states = {
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_aux_state_dict': optimizer_aux.state_dict(),
        "list_mean_losses": list_mean_losses,
        "list_mean_losses_validation": list_mean_losses_validation,
        "best_loss": best_loss,
    }
    if not args.no_scheduler:
        states["scheduler_state_dict"] = scheduler.state_dict()
        states["scheduler_aux_state_dict"] = scheduler_aux.state_dict()
    save_checkpoint(states, is_best, args.out_dir, "final.pth")



if __name__ == '__main__':
    time_start = time.time()
    main()
    training_time = time.time() - time_start
    print(f"Total time: {common_utils.format_seconds(training_time)}")
