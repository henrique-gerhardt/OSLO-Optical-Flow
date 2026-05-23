import abc
import io
import os
import platform
import subprocess
import sys
import time

from tempfile import mkstemp
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image as Image
import torch
import healpy as hp

import utils.common_function as common_utils

from pytorch_msssim import ms_ssim

from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb

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


def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)

def sampleHEALPix(hp_img, colat, lon):
    hp_img = hp_img.reshape((hp_img.shape[0], -1))
    nside = hp.npix2nside(hp_img.shape[0])

    pix, weights = hp.get_interp_weights(nside, colat, lon, nest=True, lonlat=False)
    values = np.expand_dims(weights[0], axis=1) * hp_img[pix[0], :] + \
             np.expand_dims(weights[1], axis=1) * hp_img[pix[1], :] + \
             np.expand_dims(weights[2], axis=1) * hp_img[pix[2], :] + \
             np.expand_dims(weights[3], axis=1) * hp_img[pix[3], :]

    values = np.around(values)
    assert (values >= 0).all() and (values <= 255).all()
    values = values.astype(np.uint8)
    return values

def sample_sphere(img, projection, colat, lon):
    if projection == "erp":
        samples = common_utils.sampleEquirectangular(img, colat, lon, flip=True, interpolation="bilinear")
    elif projection == "healpix":
        samples = sampleHEALPix(img, colat, lon)
    else:
        raise ValueError("Projection not defined")
    return samples

def _compute_psnr(a, b, max_val: float = 255.0, weight = None) -> float:
    mse = (a - b) ** 2
    if weight is not None:
        assert np.isclose(weight.sum(), 1)
        mse_numel = mse.numel()
        mse *= weight
        mse = mse.sum()
        if weight.numel() != mse_numel: # averaging channels when broadcasting happens
            mse /= (mse_numel / weight.numel())
    else:
        mse = torch.mean(mse)
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse.item())
    return psnr

def _compute_ms_ssim(a, b, max_val: float = 255.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()

def _compute_s_psnr(a, b, max_val: float = 255.0, projection: str = 'erp', sphere_points: str = os.path.join('misc', 'sphere_655362.txt')) -> float:
    colat_lon = np.loadtxt(sphere_points, dtype=np.float64)
    colat_lon += np.array([90, 180], dtype=colat_lon.dtype)
    colat_lon = np.radians(colat_lon)
    colat, lon = colat_lon[:, 0], colat_lon[:, 1]
    # torch (1, C, H, W) -> numpy (H, W, C)
    a = np.array(a.squeeze(0).permute(1,2,0))
    b = np.array(b.squeeze(0).permute(1,2,0))
    # resample to same uniform sampling
    a = torch.from_numpy(sample_sphere(a, projection, colat, lon)) # could be extended for different input projections
    b = torch.from_numpy(sample_sphere(b, projection, colat, lon))
    spsnr = _compute_psnr(a, b, max_val)
    return spsnr

def _compute_ws_psnr(a, b, max_val: float = 255.0, projection: str = 'erp') -> float:
    if projection == "erp":
        height = a.shape[2]
        weights = torch.arange(0, height, dtype=torch.float64)
        weights = torch.cos((weights - height/2. + 0.5) * (np.pi/height))
        weights = torch.tile(weights.reshape(-1, 1), (1, a.shape[3]))
        weights /= weights.sum()
        wspsnr = _compute_psnr(a, b, max_val, weights[None,None,:,:])
    elif projection == "healpix":
        wspsnr = _compute_psnr(a, b, max_val)
    else:
        raise ValueError("Projection not defined")
    return wspsnr

_metric_functions = {
    "psnr-rgb": _compute_psnr,
    "ms-ssim-rgb": _compute_ms_ssim,
    "s-psnr-rgb": _compute_s_psnr,
    "ws-psnr-rgb": _compute_ws_psnr,
}


def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    metrics: Optional[List[str]] = None,
    max_val: float = 255.0,
) -> Dict[str, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`."""

    if metrics is None:
        metrics = ["psnr-rgb"]

    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x

    a = _convert(a)
    b = _convert(b)

    out = {}
    for metric_name in metrics:
        out[metric_name] = _metric_functions[metric_name](a, b, max_val)
    return out


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4]


class Codec(abc.ABC):
    """Abstract base class"""

    _description = None

    def __init__(self, args):
        self._set_args(args)

    def _set_args(self, args):
        return args

    @classmethod
    @abc.abstractmethod
    def setup_args(cls, _parser):
        pass

    @property
    def description(self):
        return self._description

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    @abc.abstractmethod
    def _run_impl(self, img, quality, patch_size, *args, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        in_filepath,
        quality: int,
        metrics: Optional[List[str]] = None,
        patch_size: tuple = None,
        return_rec: bool = False,
    ):
        info, rec = self._run_impl(in_filepath, quality, patch_size)
        info.update(compute_metrics(rec, self._load_img(in_filepath), metrics))
        info['img_name'] = in_filepath.stem
        if return_rec:
            return info, rec
        return info


class PillowCodec(Codec):
    """Abstract codec based on Pillow bindings."""

    fmt = None

    @property
    def name(self):
        raise NotImplementedError()

    @classmethod
    def setup_args(cls, _parser):
        pass

    def _run_impl(self, in_filepath, quality, patch_size=None):
        img = self._load_img(in_filepath)
        w, h = img.size
        if patch_size is not None:
            w_p, h_p = patch_size
            n_patches = int(np.ceil(w / w_p) * np.ceil(h / h_p))
        else:
            n_patches = 1
            w_p, h_p = w, h
        n_patches_x = int(np.ceil(w / w_p))
        bytes, enc_time, dec_time = 0, 0, 0
        rec = np.empty((h,w,3), dtype=np.uint8)
        for patch_id in range(n_patches):
            # patch indices in x and y directions
            id_x, id_y = patch_id % n_patches_x, patch_id // n_patches_x
            x_start, x_end = id_x*w_p, min((id_x+1)*w_p, w) # end position might be outside of the image
            y_start, y_end = id_y*h_p, min((id_y+1)*h_p, h)
            img_crop = img.crop((x_start, y_start, x_end, y_end))
            # encode patch
            start = time.time()
            tmp = io.BytesIO()
            img_crop.save(tmp, format=self.fmt, quality=int(quality))
            enc_time += (time.time() - start)
            tmp.seek(0)
            size = tmp.getbuffer().nbytes
            # decode patch and save entries
            start = time.time()
            rec_crop = Image.open(tmp)
            rec_crop.load()
            dec_time += (time.time() - start)
            rec[y_start:y_end, x_start:x_end, :] = np.array(rec_crop)
            
            bytes += size

        bpp_val = bytes * 8 / (w * h)
        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }
        rec = Image.fromarray(rec)
        return out, rec


class JPEG(PillowCodec):
    """Use libjpeg linked in Pillow"""

    fmt = "jpeg"
    _description = f"JPEG. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "JPEG"


class WebP(PillowCodec):
    """Use libwebp linked in Pillow"""

    fmt = "webp"
    _description = f"WebP. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "WebP"


class BinaryCodec(Codec):
    """Call an external binary."""

    fmt = None

    @property
    def name(self):
        raise NotImplementedError()

    @classmethod
    def setup_args(cls, _parser):
        pass
    
    def _run_impl(self, in_filepath, quality, patch_size=None):
        fd0, png_filepath = mkstemp(suffix=".png")
        fd1, out_filepath = mkstemp(suffix=self.fmt)

        # Encode
        start = time.time()
        run_command(self._get_encode_cmd(in_filepath, quality, out_filepath))
        enc_time = time.time() - start
        size = filesize(out_filepath)

        # Decode
        start = time.time()
        run_command(self._get_decode_cmd(out_filepath, png_filepath))
        dec_time = time.time() - start

        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        os.close(fd1)
        os.remove(out_filepath)

        img = self._load_img(in_filepath)
        bpp_val = float(size) * 8 / (img.size[0] * img.size[1])

        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        return out, rec

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        raise NotImplementedError()

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        raise NotImplementedError()


class JPEG2000(BinaryCodec):
    """Use ffmpeg version.
    (Not built-in support in default Pillow builds)
    """

    fmt = ".jp2"

    @property
    def name(self):
        return "JPEG2000"

    @property
    def description(self):
        return f"JPEG2000. ffmpeg version {_get_ffmpeg_version()}"

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        cmd = [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-y",
            "-i",
            in_filepath,
            "-vcodec",
            "jpeg2000",
            "-pix_fmt",
            "yuv444p",
            "-c:v",
            "libopenjpeg",
            "-compression_level",
            quality,
            out_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = ["ffmpeg", "-loglevel", "panic", "-y", "-i", out_filepath, rec_filepath]
        return cmd


class BPG(BinaryCodec):
    """BPG from Fabrice Bellard."""

    fmt = ".bpg"

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-m",
            choices=["420", "444"],
            default="444",
            help="subsampling mode (default: %(default)s)",
        )
        parser.add_argument(
            "-b",
            choices=["8", "10"],
            default="8",
            help="bitdepth (default: %(default)s)",
        )
        parser.add_argument(
            "-c",
            choices=["rgb", "ycbcr"],
            default="ycbcr",
            help="colorspace  (default: %(default)s)",
        )
        parser.add_argument(
            "-e",
            choices=["jctvc", "x265"],
            default="x265",
            help="HEVC implementation (default: %(default)s)",
        )
        parser.add_argument("--encoder-path", default="bpgenc", help="BPG encoder path")
        parser.add_argument("--decoder-path", default="bpgdec", help="BPG decoder path")

    def _set_args(self, args):
        args = super()._set_args(args)
        self.color_mode = args.c
        self.encoder = args.e
        self.subsampling_mode = args.m
        self.bitdepth = args.b
        self.encoder_path = args.encoder_path
        self.decoder_path = args.decoder_path
        return args

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 0 <= int(quality) <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd


class TFCI(BinaryCodec):
    """Tensorflow image compression format from tensorflow/compression"""

    fmt = ".tfci"
    _models = [
        "bmshj2018-factorized-mse",
        "bmshj2018-hyperprior-mse",
        "mbt2018-mean-mse",
    ]

    @property
    def description(self):
        return "TFCI"

    @property
    def name(self):
        return f"{self.model}"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-m",
            "--model",
            choices=cls._models,
            default=cls._models[0],
            help="model architecture (default: %(default)s)",
        )
        parser.add_argument(
            "-p",
            "--path",
            required=True,
            help="tfci python script path (default: %(default)s)",
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.model = args.model
        self.tfci_path = args.path
        return args

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 1 <= quality <= 8:
            raise ValueError(f"Invalid quality value: {quality} (1, 8)")
        cmd = [
            sys.executable,
            self.tfci_path,
            "compress",
            f"{self.model}-{quality:d}",
            in_filepath,
            out_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [sys.executable, self.tfci_path, "decompress", out_filepath, rec_filepath]
        return cmd


def get_vtm_encoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


def get_vtm_decoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


class VTM(Codec):
    """VTM: VVC reference software"""

    fmt = ".bin"

    @property
    def description(self):
        return "VTM"

    @property
    def name(self):
        return "VTM"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="VTM build dir",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="VTM config file",
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.encoder_path = get_vtm_encoder_path(args.build_dir)
        self.decoder_path = get_vtm_decoder_path(args.build_dir)
        self.config_path = args.config
        self.rgb = args.rgb
        return args

    def _run_impl(self, in_filepath, quality, patch_size=None):
        if not 0 <= quality <= 63:
            raise ValueError(f"Invalid quality value: {quality} (0,63)")

        # Taking 8bit input for now
        bitdepth = 8

        # Convert input image to yuv 444 file
        arr = np.asarray(self._load_img(in_filepath))
        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2**bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2**bitdepth - 1)).astype(np.uint8)

        h, w = arr.shape[1:]
        if patch_size is not None:
            w_p, h_p = patch_size
            n_patches = int(np.ceil(w / w_p) * np.ceil(h / h_p))
        else:
            n_patches = 1
            w_p, h_p = w, h
        n_patches_x = int(np.ceil(w / w_p))
        bytes, enc_time, dec_time = 0, 0, 0
        rec = np.empty((3,h,w), dtype=np.uint8)
        for patch_id in range(n_patches):
            id_x, id_y = patch_id % n_patches_x, patch_id // n_patches_x
            x_start, x_end = id_x*w_p, min((id_x+1)*w_p, w) # end position might be outside of the image
            y_start, y_end = id_y*h_p, min((id_y+1)*h_p, h)
            arr_crop = np.array(arr[:, y_start:y_end, x_start:x_end])
            
            fd, yuv_path = mkstemp(suffix=".yuv")
            out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
            with open(yuv_path, "wb") as f:
                f.write(arr_crop.tobytes())
            
            # Encode
            cmd = [
                self.encoder_path,
                "-i",
                yuv_path,
                "-c",
                self.config_path,
                "-q",
                quality,
                "-o",
                "/dev/null",
                "-b",
                out_filepath,
                "-wdt",
                x_end - x_start,
                "-hgt",
                y_end - y_start,
                "-fr",
                "1",
                "-f",
                "1",
                "--InputChromaFormat=444",
                "--InputBitDepth=8",
                "--ConformanceWindowMode=1",
            ]

            if self.rgb:
                cmd += [
                    "--InputColourSpaceConvert=RGBtoGBR",
                    "--SNRInternalColourSpace=1",
                    "--OutputInternalColourSpace=0",
                ]
            
            start = time.time()
            run_command(cmd)
            enc_time += (time.time() - start)
            # cleanup encoder input
            os.close(fd)
            os.unlink(yuv_path)

            # Decode
            cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
            if self.rgb:
                cmd.append("--OutputInternalColourSpace=GBRtoRGB")
            start = time.time()
            run_command(cmd)
            dec_time += (time.time() - start)
            rec_arr_crop = np.fromfile(yuv_path, dtype=np.uint8)
            rec_arr_crop = rec_arr_crop.reshape(arr_crop.shape)

            rec_arr_crop = rec_arr_crop.astype(np.float32) / (2**bitdepth - 1)
            if not self.rgb:
                # arr_crop = ycbcr2rgb(torch.from_numpy(arr_crop.copy())).numpy()
                rec_arr_crop = ycbcr2rgb(torch.from_numpy(rec_arr_crop.copy())).numpy()

            rec[:, y_start:y_end, x_start:x_end] = rec_arr_crop
            
            bytes += filesize(out_filepath)

            # Cleanup
            os.unlink(yuv_path)
            os.unlink(out_filepath)

        bpp_val = bytes * 8.0 / (h * w)
        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        rec = Image.fromarray(
            (rec.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        
        return out, rec


class HM(Codec):
    """HM: H.265/HEVC reference software"""

    fmt = ".bin"

    @property
    def description(self):
        return "HM"

    @property
    def name(self):
        return "HM"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="HM build dir",
        )
        parser.add_argument(
            "-c", "--config", type=str, required=True, help="HM config file"
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.encoder_path = os.path.join(args.build_dir, "TAppEncoderStatic")
        self.decoder_path = os.path.join(args.build_dir, "TAppDecoderStatic")
        self.config_path = args.config
        self.rgb = args.rgb
        return args

    def _run_impl(self, in_filepath, quality, patch_size=None):
        if not 0 <= quality <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")

        # Convert input image to yuv 444 file
        arr = np.asarray(self._load_img(in_filepath))
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
        bitdepth = 8

        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2**bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2**bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=8",
            "--SEIDecodedPictureHash",
            "--Level=5.1",
            "--CUNoSplitIntraACT=0",
            "--ConformanceMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        start = time.time()

        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]

        if self.rgb:
            cmd.append("--OutputInternalColourSpace=GBRtoRGB")

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start
        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)
        arr = arr.astype(np.float32) / (2**bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)
        if not self.rgb:
            arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        bpp = filesize(out_filepath) * 8.0 / (height * width)

        # Cleanup
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        return out, rec


class AV1(Codec):
    """AV1: AOM reference software"""

    fmt = ".webm"

    @property
    def description(self):
        return "AV1"

    @property
    def name(self):
        return "AV1"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="AOM binaries dir",
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.encoder_path = os.path.join(args.build_dir, "aomenc")
        self.decoder_path = os.path.join(args.build_dir, "aomdec")
        return args

    def _run_impl(self, img, quality, patch_size=None):
        if not 0 <= quality <= 63:
            raise ValueError(f"Invalid quality value: {quality} (0,63)")

        # Convert input image to yuv 444 file
        arr = np.asarray(img)
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".webm"
        bitdepth = 8

        arr = arr.transpose((2, 0, 1))  # color channel first

        # convert rgb content to YCbCr
        rgb = torch.from_numpy(arr.copy()).float() / (2**bitdepth - 1)
        arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
        arr = (arr * (2**bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-w",
            width,
            "-h",
            height,
            "--fps=1/1",
            "--limit=1",
            "--input-bit-depth=8",
            "--cpu-used=0",
            "--threads=1",
            "--passes=2",
            "--end-usage=q",
            "--cq-level=" + str(quality),
            "--i444",
            "--skip=0",
            "--tune=psnr",
            "--psnr",
            "--bit-depth=8",
            "-o",
            out_filepath,
            yuv_path,
        ]

        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [
            self.decoder_path,
            out_filepath,
            "-o",
            yuv_path,
            "--rawvideo",
            "--output-bit-depth=8",
        ]

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        arr = arr.astype(np.float32) / (2**bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)

        arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
        rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        bpp = filesize(out_filepath) * 8.0 / (height * width)

        # Cleanup
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )
        return out, rec
