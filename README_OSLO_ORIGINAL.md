## Original OSLO README

## Introduction
This is the PyTorch implementation of our paper "**OSLO: On-the-Sphere Learning for Omnidirectional images and its application to 360-degree image compression**" accepted in **IEEE Transactions on Image Processing** with the extensions of out paper "**OSLO-IC: On-the-Sphere Learned Omnidirectional Image Compression with Attention Modules and Spatial Context**" accepted for **ICASSP 2025**. In case of using this code, please cite our work as mentioned in section [Citation](#citation). Some pretrained models and a few 360-degree images are provided in [oslo_data](https://gitlab.inria.fr/on-the-sphere-learning-for-omnidirectional-images-oslo/oslo_data.git). Note that in our paper we used  SUN360 equirectangular image database of size 9104×4552 for experiments, but the provided images in **oslo_data** repository are different from the SUN360 images.


## Prerequisites
The code is based on **PyTorch**. So, first you should install **PyTorch** in your virtual environment depending on your **cuda version** (if you don't have cuda, use the cpu version of PyTorch). You also need to install [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) depending again on your **cuda and PyTorch versions**. The other dependencies can be installed using the following command

```
pip install -r requirements.txt
```

 
## <a id="healpix-sampling"></a> HEALPix sampling 
HEALPix (an acronym for Hierarchical Equal Area isoLatitude Pixelation) is an algorithm for pixelisation of the 2-sphere in which each pixel covers the same surface area as every other pixel. In HEALPix, the tessellation process begins by partitioning the spherical surface into 12 equal-area regions (base resolution). To have finer pixelization, each region is recursively divided into 2x2 equal-area sub-pixels until the desired resolution is reached.

A Healpix tessellation is parametrized by a number, called $N_\mathrm{side}=2^r$, and the total number of pixels is equal to $12\cdot N_\mathrm{side}^2=12\cdot 4^r$ with resolution $r$.

By construction, each pixel in HEALPix has eight adjacent neighbors (structured in a diamond pattern), except for 24 pixels that have seven neighbors located at the intersection of polar base tiles with equatorial tiles (for any resolution higher than the base resolution). The orientation of neighboring points with the central point is almost fixed all over the sphere. Each neighbor can be identified by its relative direction to the central pixel: SW, W, NW, N, NE, E, SE, and S. 

## <a id="step-1"></a> Step 1: Sampling from Equirectangular Images
In the first step, the equirectangular images must be sampled according to HEALPix sampling. For that, use `generate_healpix_samples_and_split_to_train_and_test.py`. The script samples the sphere and at the same time it splits the dataset into train, validation, and test sets. For example:

```
python ./generate_healpix_samples_and_split_to_train_and_test.py -i ../oslo_data/images_equirectangular -e jpg -r 10 -s -o ../oslo_data/Healpix_res_10 -t 0.8 -v 0.1
```
The script generates the sampled files in numpy `.npy` format. It also generates three text files named `train.txt`, `test.txt`, `validation.txt`.

**Arguments description**:


* `[-i|--image-dir]`: Directory of equirectangular images
* `[-e|--image-ext]`: Image type (accepted values are ["jpg", "jpeg", "png", "tiff"])
* `[-r|--healpix-res]`: Resolution of HEALPix for sampling (see [HEALPix sampling](#healpix-sampling)).
* `[-s|--run-healpix-sampling]`: Runs HEALPix sampling. Otherwise, it only splits the dataset into train/test/validation sets).
* `[-o|--out-dir]`: Output directory where the healpix sampling are stored in numpy arrays.
* `[-t|--ratio-train-data]`: Ratio of data that will be considered as training data.
* `[-v|--ratio-validation-data]`: Ratio of data that will be considered as validation data.

## <a id="step-2"></a> Step 2: Neighboring Structure
Now you need to create a structure that determines the neighboring of each pixel. Each pixel in HEALPix has eight adjacent neighbors and each neighbor can be identified by its relative direction to the central pixel: SW, W, NW, N, NE, E, SE, and S. This structure can be defined using `generate_and_save_healpix_oslo_struct.py`. For example:

```
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 10 -pr 8 -nh 1
```

**Arguments description**:

* `[-o|--out-dir]`: Directory to save structures.
* `[-hr|--healpix-res]`: Resolution of HEALPix sampling.
* `[-pr|--patch-res]`: Resolution of the HEALPix patches.
* `[-nh|--num-hops]`: Neighborhood size (n-hop neighborhood). Default is 1. If >1, use `[-gd|--grid-dir]` to define the directory path containing grid files (can be downloaded from [neighbor_grids](https://gitlab.inria.fr/on-the-sphere-learning-for-omnidirectional-images-oslo/oslo_data/-/tree/main/neighbor_grids?ref_type=heads)).

**Note**: In our paper we used the following values for the pair `(-hr, -pr)`: `{(10, 8), (9, 7), (8, 6), (7, 5), (6, 4), (5, 3), (4, 2)}`. For the fist element in the set, i.e., `(10, 8)`, `hr=10` determines the input resolution of HEALPix images which is equivalent to $N_\mathrm{pix}=12\cdot 4^{10}=12,582,912$ pixels (a resolution almost equal to 4K for 2D equirectangular images), and `pr=8`determines the patch resolution of $2^8\times 2^8=256\times 256$ as it is suggested in [[Ballé2017]](#balle2017). The remaining pairs in the set refer to downsampled healpix resolution and their corresponding patch size used in auto-encoder architectures explained in [[Ballé2017]](#balle2017), [[Ballé2018]](#balle2018). So, to construct the rest we do:

```
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 9 -pr 7
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 8 -pr 6
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 7 -pr 5
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 6 -pr 4
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 5 -pr 3
python ./generate_and_save_healpix_oslo_struct.py -o ../oslo_data/neighbor_structure -hr 4 -pr 2
```

**Side Note**: If you want to create DeepSphere graph structure as explained in [[Perraudin2019]](#perraudin2019), which results in less expressive isotropic convolutional filters, you can use script `generate_and_save_healpix_deepsphere_graph.py` which has almost similar arguments.

## Step 3: Training
The training is done using `main_sphere_compression.py` script. Three architectures were implemented in this script: Factorized prior [[Ballé2017]](#balle2017), scale hyperprior [[Ballé2018]](#balle2018) and mean and scale hyperprior [[Minnen2018]](#minnen2018) with optional spatial context model [[Minnen2018]](#minnen2018), attention modules [[Cheng2020]](#cheng2020) and residual blocks instead of GDN/IGDN layers [[He2022]](#he2022). For example, to use mean and scale hyperprior architecture with attention modules, context model, residual blocks as nonlinearities, and transposed convolution as unpooling, one can use it as follows:

```
python ./main_sphere_compression.py --gpu -m SphereMeanScaleHyperprior -at -cm autoregressive -nl RB -upf SDPAConvTransposed -nd ../oslo_data/neighbor_structure -i ../oslo_data/Healpix_res_10/train.txt -o ../oslo_data/meanScaleHyperprior_lambda_0.0018 --lambda 0.0018 -q 1 --dataloader-num-workers 2 -bst 10 --checkpoint-interval 10
```

**Note**: If you get `segmentation fault (core dumped)` while running the training, please double check that the torch version is compatible with `torch_geometric` ([PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)) version.

**Arguments description**:

* `[-g|--gpu]`: Run on with GPU (cuda). If GPU is not available remove this argument.
* `[-m|--model]`: Architecture to use which must be one of: **SphereFactorizedPrior**, **SphereFactorizedPrior** or **SphereMeanScaleHyperprior**.
* `[-at|--attention]`: Use attention modules.
* `[-cm|--context-model]`: Which context model to use. If set, must be either **autoregressive** (standard) or **full** (not masked, thus not causal and just for reference).
* `[-nl|--nonlinearity]`: Which nonlinearity to use. Must be either **GDN** for (inverse) generalized divisive normalization or **RB** for residual blocks.
* `[-upf|--unpool-func]`: Which unpool function to use. Must be one of: **pixel_shuffle** (Default, baseline implementation), **SDPAConvTransposed** (recommended, lower number of trainable parameters), **nearest**, **linear**, **bilinear**, **bicubic**, **trilinear**.
* `[-i|--train-data]`: Path of the text file containing location of train images in HEALPix format (Created in [step 1](#step-1)).
* `[-nd|--neighbor-struct-dir]`: Path to the neighboring structure (Created in [step 2](#step-2)).
* `[-o|--out-dir]`: Path to the output directory to save model and results.
* `[--lambda]`: Bit-rate distortion parameter in "Rate + lambda . Distortion".
* `[-q|--quality]`: Quality level that determines number of parameters in the architecture (1: lowest, highest: 8)
* `[--dataloader-num-workers]`: Determines the number of loader worker processes for multi-process data loading.
* `[-bst|--batch-size-train]`: Batch size for training.
* `[--checkpoint-interval]`: Interval to be used for saving data either for inference or resuming training.
* `[--no-scheduler]`: Disable scheduler

You can define a validation set during training to have an scheduler to reduce the learning rate based on the loss value on the validation set.

```
python ./main_sphere_compression.py --gpu -m SphereMeanScaleHyperprior -at -cm autoregressive -nl RB -upf SDPAConvTransposed -nd ../oslo_data/neighbor_structure -i ../oslo_data/Healpix_res_10/train.txt -v ../oslo_data/Healpix_res_10/validation.txt --validation-start 801 -o ../oslo_data/meanScaleHyperprior_lambda_0.0018 --lambda 0.0018 -q 1 --dataloader-num-workers 2 -bst 10 -bsvt 10 --checkpoint-interval 1 --scheduler-patience 10
``` 

**Additional Arguments**:

* `[-v|--validation-data]`: Path to the text file containing location of the validation images in HEALPix format (Created in [step 1](#step-1)).
* `[--validation-start]`: Epoch, in which to start validation. Default is 801 for 1000 epochs (first 800 epochs will have constant learning rate).
* `[-bsvt|--batch-size-valtest]`: Batch size for validation and test datasets.
* `[--scheduler-patience]`: Patience in terms of number of epochs for scheduler ReduceLROnPlateau.

**Note**: If you want to load a checkpoint model for inference or resuming training you can use the following argument:

* `[--checkpoint-file]`: File address to resume training from the previous saved checkpoint

For example, to load the checkpoint of epoch 800:

```
python ./main_sphere_compression.py --gpu -m SphereMeanScaleHyperprior -at -cm autoregressive -nl RB -upf SDPAConvTransposed -nd ../oslo_data/neighbor_structure -i ../oslo_data/Healpix_res_10/train.txt -v ../oslo_data/Healpix_res_10/validation.txt -o ../oslo_data/meanScaleHyperprior_lambda_0.0018 --lambda 0.0018 -q 1 --dataloader-num-workers 2 -bst 10 -bsvt 10 --checkpoint-interval 1 --scheduler-patience 10 --checkpoint-file ../oslo_data/meanScaleHyperprior_lambda_0.0018/checkpoint_800.pth.tar
``` 

**Side note**:  The script can be used to compress images using DeepSphere architecture [[Perraudin2019]](#perraudin2019), which results in less expressive isotropic convolutional filters due to the use of graph convolution. To compress a HEALPix image using DeepSphere, one can use the following script with arguments `-pf max_pool -upf pixel_shuffle -c ChebConv -w gaussian`:

```
python ./main_sphere_compression.py --gpu -m SphereScaleHyperprior -nd ../oslo_data/neighbor_structure -i ../oslo_data/Healpix_res_10/train.txt -v ../oslo_data/Healpix_res_10/validation.txt -o ../oslo_data/DeepSphere_lambda_0.0018 --lambda 0.0018 -q 1 --dataloader-num-workers 2 -bst 10 -bsvt 10 -pf max_pool -upf pixel_shuffle -c ChebConv -w gaussian --checkpoint-interval 1 --scheduler-patience 10 --checkpoint-file ../oslo_data/DeepSphere_lambda_0.0018/checkpoint_800.pth.tar 
```
Note that to use the above-mentioned script for DeepSphere, the graph structures must be created using `generate_and_save_healpix_deepsphere_graph.py` and saved in `../oslo_data/neighbor_structure`.

## <a id="step-4"></a> Step 4: Evaluation
The same `main_sphere_compression.py` script can be used to evaluate a trained model on the test dataset. The trained model can be loaded using `[--checkpoint-file]`. For example, to evaluate a `FactorizePrior` trained model we can use:

```
python ./main_sphere_compression.py --gpu -m SphereFactorizedPrior -nd ../oslo_data/neighbor_structure -t ../oslo_data/Healpix_res_10/test.txt -o ../oslo_data/factorizedPrior_lambda_0.0932 --lambda 0.0932 -q 7 --dataloader-num-workers 2 -bsvt 10 --checkpoint-file ../oslo_data/checkpoints/factorizedPrior/factorizedPrior_lambda_0.0932_q_7.pth.tar --foldername-valtest reconstruction
``` 

**Additional Arguments**:

* `[--foldername-valtest]`: folder name (created in the address specified by `-o` argument) in which the reconstructed files are stored in `.npy` format. Note that
 
	- In addition to the reconstructed format in `.npy`, the **grayscale** Mollweide projection of the reconstructed HEALPix files are stored in `png` format for visualization. This can optionally be avoided to save memory by setting `[--only-npy-valtest]`.
	- Also, the rate of each file is written in `rates.txt`. In this file, for each image the theoretical (second column) and actual rates (third column) are written. Due to some inefficiency from the entropy coder and also some implementation details there is a slight difference between the actual and theoretical rates.


**Note**: The `lambda` and `quality` values must match with the values used during training and the checkpoint is saved with. For example, by changing `lambda` value to 0.0009, the `quality` is changed to 1 as follows:

```
python ./main_sphere_compression.py --gpu -m SphereFactorizedPrior -nd ../oslo_data/neighbor_structure -t ../oslo_data/Healpix_res_10/test.txt -o ../oslo_data/factorizedPrior_lambda_0.0009 --lambda 0.0009 -q 1 --dataloader-num-workers 2 -bsvt 10 --checkpoint-file ../oslo_data/checkpoints/factorizedPrior/factorizedPrior_lambda_0.0009_q_1.pth.tar --foldername-valtest reconstruction
``` 

**Note**: In the experiments explained in our paper, we chose the values for the pair `(lambda, quality)` from the following set: `{(0.0005, 1), (0.0009, 1), (0.0018, 1), (0.0035, 2), (0.0067, 3), (0.0130, 4), (0.0250, 5), (0.0483, 6), (0.0932, 7), (0.1800, 8)}` and skipped `{(0.0009, 1), (0.0035, 2)}` in case of 8 RD points. 

## Computing Spherical BD-Rates
Script `compute_spherical_psnr.py` is provided in `misc` folder to compute rate-spherical BD-rates. Spherical PSNR (S-PSNR) and Weighted to Spherically uniform PSNR (WS-PSNR) are implemented as the objective quality evaluation. The script must be called as follows:

```
python ./misc/compute_spherical_psnr.py --original-dir ../oslo_data/images_equirectangular --original-ext jpg --projection-original erp -t ../oslo_data/Healpix_res_10/test.txt --test-files-prefix healpix_sampling_res_10_ --reconstruction-dir ../oslo_data/factorizedPrior_lambda_0.0009 --reconstruction-subfolder reconstruction --reconstruction-ext npy --reconstruction-prefix healpix_sampling_res_10_ --reconstruction-suffix _reconstructed --projection healpix --rate-prefix healpix_sampling_res_10_ --sphere-points ./misc/sphere_655362.txt
```

**Additional Arguments**:

* `[-o|--original-dir]`: Directory where the original images are.
* `[-oe|--original-ext]`: Extension of the original images.
* `[-po|--projection-original]`: Projection type of the original images. In our case it is `erp` (equirectangular).
* `[-t|--test-data]`: The text file containing location of test images.
* `[--test-files-prefix]`: Common prefix in the uncompressed test file names (compared to file names of the original images).
* `[--test-files-suffix]`: Common suffix in the uncompressed test file names (compared to file names of the original images)..
* `[-r|--reconstruction-dir]`: Main directory where the reconstructed projections are.
* `[-rf|-reconstruction-subfolder]`: Folder name where the reconstructed projections are. It is the folder name we used after argument `--foldername-valtest` during evaluation in [step 4](#step-4) where the `rates.txt` is stored.
* `[-re|--reconstruction-ext]`: Extension of the reconstructed files.
* `[-rp|--reconstruction-prefix"]`: Common prefix in the reconstructed file names.
* `[-rs|--reconstruction-suffix"]`: Common suffix in the reconstructed file names.
* `[-p|--projection"]`: Projection of reconstructed images (`healpix` in our case).
* `[-ratef|--rate-prefix"]`: File name prefix of each reconstructed image in the rate text file.
* `[-rates|--rate-suffix"]`: File name suffix in each reconstructed image in the rate text file.
* `[-s|--sphere-points"]`: File that shows sphere points uniformly sampled (needed for SPSNR and it is stored in `misc` folder).

## Visualisation
To plot a Healpix sampled map in Mollweide projection `create_mollweide_projection.py` script is provided in `misc` folder. Note that Mollweide projection is a different projection and it is different from HEALPix sampling. It is just a way of visualization. The script can be used as follows:

```
python ./misc/create_mollweide_projection.py -i ../oslo_data/Healpix_res_10/healpix_sampling_res_10_01.npy -o ../oslo_data/Healpix_res_10/healpix_sampling_res_10_01.png -w 4000
```

**Additional Arguments**:

* `[-i|--input]`: File address of the input healpix sampled map (stored in npy file).
* `[-o|--output]`: File address indicating where to save the Mollweide projection.
* `[-w|--width]`: Width of the Mollweide projection image


## License

This code is licensed under **GNU General Public License v3.0**.

## <a id="citation"></a> Citation
If you use this project, please cite our relevant publications:

```
@ARTICLE{mahmoudian2022oslo,
  author={Mahmoudian Bidgoli, Navid and de A. Azevedo, Roberto G. and Maugey, Thomas and Roumy, Aline and Frossard, Pascal},
  journal={IEEE Transactions on Image Processing}, 
  title={OSLO: On-the-Sphere Learning for Omnidirectional Images and Its Application to 360-Degree Image Compression}, 
  year={2022},
  volume={31},
  number={},
  pages={5813-5827},
  doi={10.1109/TIP.2022.3202357},
}
@INPROCEEDINGS{10889131,
  author={Wawerek-López, Paul and Bidgoli, Navid Mahmoudian and Frossard, Pascal and Kaup, André and Maugey, Thomas},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={OSLO-IC: On-the-Sphere Learned Omnidirectional Image Compression with Attention Modules and Spatial Context}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889131},
}
```


## References
<a id="balle2017">[Ballé2017]</a> 
J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image compression,” in *International Conference on Learning Representations (ICLR)*, 2017.

<a id="balle2018">[Ballé2018]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in *International Conference on Learning Representations (ICLR)*, 2018.

<a id="perraudin2019">[Perraudin2019]</a> 
N. Perraudin, M. Defferrard, T. Kacprzak, and R. Sgier, “DeepSphere: Efficient spherical convolutional neural network with HEALPix sampling for cosmological applications,” *Astronomy and Computing*, vol. 27, pp. 130–146, Apr. 2019.

<a id="minnen2018">[Minnen2018]</a> 
D. Minnen, J. Ballé, and G. D Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, pp. 10771–10780, Dec. 2018.

<a id="cheng2020">[Cheng2020]</a>
Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Jun. 2020.

<a id="he2022">[He2022]</a>
D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding,” in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Jun. 2022.

