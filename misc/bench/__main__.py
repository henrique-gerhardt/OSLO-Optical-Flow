"""Benchmark evaluations from compressai.utils.bench with S-PSNR and WS-PSNR evaluation."""
import argparse
import json
import multiprocessing as mp
import sys, os

from collections import defaultdict
from itertools import starmap
from pathlib import Path
from typing import List
from math import nan

from .codecs import AV1, BPG, HM, JPEG, JPEG2000, TFCI, VTM, Codec, WebP

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

codecs = [JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1]


# we need the quality index (not value) to compute the stats later
def func(codec, i, *args):
    rv = codec.run(*args)
    return i, rv


def collect(
    codec: Codec,
    dataset: str,
    qps: List[int],
    metrics: List[str],
    num_jobs: int = 1,
    out_dir: str = None,
    patch_size: tuple = None,
):
    filepaths = []
    if Path(dataset).is_dir(): # take all images in the directory
        for ext in IMG_EXTENSIONS:
            filepaths.extend(Path(dataset).rglob(f"*{ext}"))
    elif Path(dataset).is_file():
        with open(dataset, "r") as f: # take images given in file
            lines = f.read().splitlines()
            root_dir = lines[1].split('/')
            root_dir = Path('' if root_dir[0] else os.sep).joinpath(*root_dir)
            img_names = lines[2:]
        for img_name in img_names:
            if img_name.endswith(IMG_EXTENSIONS):
                filepaths.append(root_dir.joinpath(img_name))
    else: # neither file nor directory
        raise OSError(f"No such directory: {dataset}")


    pool = mp.Pool(num_jobs) if num_jobs > 1 else None

    if len(filepaths) == 0:
        print("No images found in the dataset directory")
        sys.exit(1)

    args = [
        (codec, i, f, q, metrics, patch_size) for i, q in enumerate(qps) for f in sorted(filepaths)
    ]

    if pool:
        rv = pool.starmap(func, args)
    else:
        rv = list(starmap(func, args))

    results = [defaultdict(float) for _ in range(len(qps))]

    if out_dir:
        folder_suffix = f"_{patch_size[0]}x{patch_size[1]}" if patch_size else ""
        out_dir_codec = Path(out_dir).joinpath(codec.name+folder_suffix)
        if not out_dir_codec.is_dir():
            out_dir_codec.mkdir(parents=True, exist_ok=True)
        
        # list of shape (len(qp), len(images))
        rv_resort = [[] for _ in range(len(qps))]
        for i, metrics in rv:
            rv_resort[i].append(metrics)
        
        psnr_list = [
            "s-psnr-rgb",
            "ws-psnr-rgb",
            "psnr-rgb",
        ]
        nameWidth, metricWidth, rateWidth = len(filepaths[0].stem), 22, 30
        for i, (q, img_metrics) in enumerate(zip(qps, rv_resort)):
            fp = out_dir_codec.joinpath(f"rate_metrics_rec_q_{q:02d}.txt")
            if not fp.is_file(): # create file with header
                with open(fp, "w") as f:
                    fmt_metric = '{:^' + str(metricWidth) + 's}'
                    fmt_rate = '{:<' + str(rateWidth) + 's}'
                    header = "#" + "name".center(nameWidth - 1) + \
                            (fmt_metric * 4).format("SPSNR Color", "WSPSNR Color", "PSNR Color", "MSSSIM Color") +\
                            (fmt_rate).format("Rate")
                    f.write(header+"\n")

                    header_units = "#" + " " * (nameWidth - 1) + \
                            (fmt_metric * 4).format("(dB)", "(dB)", "(dB)", "") + \
                            (fmt_rate).format("(bpp)")
                    f.write(header_units+"\n")

            fmt_psnr = '{:^' + str(metricWidth) + '.2f}'
            fmt_ssim = '{:^' + str(metricWidth) + '.4f}'
            fmt_rate = '{:<' + str(rateWidth) + '}'
            # check existing files
            imgs_tested = []
            with open(fp, "r") as f:
                for line in f:
                    if line.startswith("#"): continue
                    img_name = line.split()[0] # caution: does not check if all metrics were calculated
                    imgs_tested.append(img_name)
            with open(fp, "a") as f:
                for img_fp, metrics in zip(filepaths, img_metrics):
                    img_name = img_fp.stem
                    if img_name in imgs_tested: continue
                    f.write(
                        img_name.center(nameWidth) + (fmt_psnr * len(psnr_list)).format(*[metrics.get(m, nan) for m in psnr_list]) + \
                        fmt_ssim.format(metrics.get("ms-ssim-rgb", nan)) + fmt_rate.format(metrics["bpp"]) + "\n"
                    )
    
    # aggregate results for all images
    for i, metrics in rv:
        for k, v in metrics.items():
            if isinstance(v, float):
                results[i][k] += v

    for i, _ in enumerate(results):
        for k, v in results[i].items():
            results[i][k] = v / len(filepaths)

    # list of dict -> dict of list
    out = defaultdict(list)
    for r in results:
        for k, v in r.items():
            out[k].append(v)
    return out


def setup_args():
    description = "Collect codec metrics."
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="codec", help="Select codec")
    subparsers.required = True
    return parser, subparsers


def setup_common_args(parser):
    parser.add_argument(
        "-d", "--dataset", type=str,
        help="Path to the dataset directory or text file containing the directory path and image file names",
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", type=str,
        help="Path to directory where to store rates and metrics for each image",
    )
    parser.add_argument(
        "-j", "--num-jobs", type=int, metavar="N", default=1,
        help="number of parallel jobs (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--qps", dest="qps", type=str, default="75",
        help="list of quality/quantization parameter seperated by comma (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics", dest="metrics", default=["psnr-rgb", "ms-ssim-rgb", "s-psnr-rgb", "ws-psnr-rgb"], nargs="+",
        help="choose metrics from [psnr-rgb, ms-ssim-rgb, s-psnr-rgb, ws-psnr-rgb] (use for very small images)",
    )
    parser.add_argument(
        "--patch-size", "-ps", type=int, nargs=2, default=None,
        help="The image will be split into patches of this size. Each patch will be compressed and decompressed separately (default: %(default)s, compress full image)"
    )
    


def main(argv):
    parser, subparsers = setup_args()
    for c in codecs:
        cparser = subparsers.add_parser(c.__name__.lower(), help=f"{c.__name__}")
        setup_common_args(cparser)
        c.setup_args(cparser)
    args = parser.parse_args(argv)

    codec_cls = next(c for c in codecs if c.__name__.lower() == args.codec)
    codec = codec_cls(args)
    qps = [int(q) for q in args.qps.split(",") if q]
    results = collect(
        codec,
        args.dataset,
        sorted(qps),
        args.metrics,
        args.num_jobs,
        args.out_dir,
        args.patch_size,
    )

    output = {
        "name": codec.name,
        "description": codec.description,
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
