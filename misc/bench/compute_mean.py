"""Append mean values to results txt file."""
import argparse
import sys

from pathlib import Path
from typing import List
import numpy as np

from .codecs import AV1, BPG, HM, JPEG, JPEG2000, TFCI, VTM, Codec, WebP

codecs = [JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1]


def append_mean(out_dir:str, codec_name:str, qps:List[int], patch_size=None):
    dirname = codec_name
    if patch_size is not None:
        w, h = patch_size
        dirname += f"_{w}x{h}"
    out_dir_codec = Path(out_dir).joinpath(dirname)
    for q in qps:
        fp = out_dir_codec.joinpath(f"rate_metrics_rec_q_{q:02d}.txt")  
        results = np.loadtxt(fp, usecols=range(1,6))
        mean = results.mean(axis=0, where=~np.isnan(results))
        
        with open(fp, "r") as f:
            lines = f.readlines()
        
        nameWidth = len(lines[-1].split()[0])
        metricWidth, rateWidth = 22, 30
        fmt_psnr = '{:^' + str(metricWidth) + '.2f}'
        fmt_ssim = '{:^' + str(metricWidth) + '.4f}'
        fmt_rate = '{:<' + str(rateWidth) + '}'
        
        line = "Average".center(nameWidth) + (fmt_psnr * 3).format(*mean[:3]) + \
            fmt_ssim.format(mean[3]) + fmt_rate.format(mean[-1]) + "\n"
        
        with open(fp, "a") as f:
            f.write("-"*len(line)+"\n") # before that print a line of "-"
            f.write(line)


def setup_args(parser):
    parser.add_argument(
        "-c",
        "--codec",
        type=str,
        help="Select codec from [jpeg, webp, jpeg2000, bpg, tfci, vtm, hm, av1]",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        help="Path to directory where to store rates and metrics for each image",
    )
    parser.add_argument(
        "-q",
        "--qps",
        type=str,
        default="75",
        help="list of quality/quantization parameter seperated by comma (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size", "-ps", type=int, nargs=2, default=None,
        help="The image will be split into patches of size (width, height). Each patch will be compressed and decompressed separately (default: %(default)s, compress full image)"
    )
    return parser

def main(argv):
    parser = argparse.ArgumentParser()
    parser = setup_args(parser)
    args = parser.parse_args(argv)

    codec_name = next(c.__name__ for c in codecs if c.__name__.lower() == args.codec)
    qps = [int(q) for q in args.qps.split(",") if q]
    append_mean(args.out_dir, codec_name, qps, args.patch_size)


if __name__=="__main__":
    main(sys.argv[1:])