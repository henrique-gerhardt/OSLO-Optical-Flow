import os, sys, inspect
current_frame = inspect.currentframe()
if current_frame is not None:
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(current_frame)))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
import glob
import random
import utils.common_function as commonFunc_utils
import healpy as hp
import utils.healpix as hp_utils
import numpy as np
import argparse
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # To solve the problem on the GRID



parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", "-i", type=str, default="./", help="Directory where the images are.")
parser.add_argument("--image-ext", "-e", type=str, default="jpg", choices=["jpg", "jpeg", "png", "tiff"], help="Extension of images.")
parser.add_argument("--healpix-res", "-r", type=int, default=10, help="Resolution of the healpix for sampling.")
parser.add_argument("--run-healpix-sampling", "-s", action="store_true", help="Run healpix sampling.")
parser.add_argument("--keep-existing", action="store_true", help="Don't overwrite already sampled images.")
parser.add_argument("--no-set-separation", action="store_true", help="Do not create text file to split train/test/validation sets")
parser.add_argument("--out-dir", "-o", type=str, default="./", help="Output directory where the healpix sampling are stored in numpy arrays.")
parser.add_argument("--ratio-train-data", "-t", type=float, default=0.75, help="Ratio of data that will be considered as training data.")
parser.add_argument("--ratio-validation-data", "-v", type=float, help="Ratio of data that will be considered as validation data.")
parser.add_argument("--interpolation", type=str, default="bilinear", help="Interpolation to use")
args = parser.parse_args()

def main(argv):
    args = parser.parse_args(argv)
    print("=========== printing args ===========")
    for key, val in args.__dict__.items():
        print(f"{key:_<40}: {val}\n")  # check this for all kinds of formatting
    print("=" * 60 + "\n")

#    assert args.no_set_separation == args.run_healpix_sampling, "if no-set-separation is set, then run-healpix-sampling must be set"
    img_dir = os.path.abspath(args.image_dir)
    img_ext = args.image_ext
    healpix_res = args.healpix_res
    run_healpixSampling = args.run_healpix_sampling
    out_dir = os.path.abspath(args.out_dir)	# Relative to absolute path
    ratio_train = args.ratio_train_data
    if args.ratio_validation_data:
        ratio_validation = args.ratio_validation_data
        assert ratio_train+ratio_validation <= 1., "ratio validation and ratio train can not exceed 1."
    else:
        ratio_validation = 0.

    nside = hp.order2nside(healpix_res)  # == 2 ** sampling_resolution
    list_files = glob.glob(img_dir + '/**/*' + img_ext, recursive=True)
    random.shuffle(list_files) # shuffling the file list
    os.makedirs(out_dir, exist_ok=True)
    print(f"write files in {out_dir}")


    list_np_filename = [None] * len(list_files)

    for i, file_path in enumerate(list_files):
        img_filename = os.path.splitext(os.path.basename(file_path))[0]
        np_filename = f"healpix_sampling_res_{healpix_res}_"+img_filename+".npy"
        list_np_filename[i] = np_filename
        np_fileAddress = os.path.join(out_dir, np_filename)
        if run_healpixSampling:
            if args.keep_existing and os.path.isfile(np_fileAddress):
                print(f'[{i+1:04n}/{len(list_files):04n}] HEALPix sampling exists of image:', img_filename, flush=True)
            else:
                print(f'[{i+1:04n}/{len(list_files):04n}] HEALPix sampling of image:', img_filename, flush=True)
                I = Image.open(file_path)
                colors = hp_utils.sampleEquirectangularForHEALPix(np.array(I), nside, interpolation=args.interpolation, nest=True)
                np.save(np_fileAddress, colors)
        else:
            assert os.path.isfile(np_fileAddress), f"file {np_fileAddress} does not exist"

        # print(f"file[{i}]={file_path} \t -> {img_filename} \t -> {np_fileAddress}")

    print(list_np_filename)
    if not args.no_set_separation:
        # the files in list_np_filename is already shuffled in random.shuffle(list_files)
        n_train = round(ratio_train * len(list_np_filename))
        n_validation = round(ratio_validation * len(list_np_filename))
        print(f"# train={n_train}, #validation={n_validation}, # test={len(list_np_filename)-(n_train+n_validation)}, # total={len(list_np_filename)}")
        list_train_filename = list_np_filename[:n_train]
        list_validation_filename  = list_np_filename[n_train:n_train+n_validation]
        list_test_filename = list_np_filename[n_train + n_validation:]

        print("Generating train set ...")
        with open(os.path.join(out_dir, "train.txt"), 'w') as f:
            f.write("#This line is a comment. The first line after this comment is the healpix resolution used for sampling. The second line is the root directory where the files are located. Then, all file names are listed\n")
            f.write(str(healpix_res)+"\n")
            f.write(out_dir+"\n")
            for item in list_train_filename:
                f.write("%s\n" % item)

        print("Generating Validation set ...")
        with open(os.path.join(out_dir, "validation.txt"), 'w') as f:
            f.write("#This line is a comment. The first line after this comment is the healpix resolution used for sampling. The second line is the root directory where the files are located. Then, all file names are listed\n")
            f.write(str(healpix_res)+"\n")
            f.write(out_dir+"\n")
            for item in list_validation_filename:
                f.write("%s\n" % item)

        print("Generating Test set ...")
        with open(os.path.join(out_dir, "test.txt"), 'w') as f:
            f.write("#This line is a comment. The first line after this comment is the healpix resolution used for sampling. The second line is the root directory where the files are located. Then, all file names are listed\n")
            f.write(str(healpix_res)+"\n")
            f.write(out_dir+"\n")
            for item in list_test_filename:
                f.write("%s\n" % item)


if __name__ == "__main__":
    main(sys.argv[1:])
