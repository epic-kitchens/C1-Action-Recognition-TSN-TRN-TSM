# EPIC-KITCHENS Action Recognition baselines

> Train/Val/Test splits and annotations are available at
> [Annotations Repo](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

> To participate and submit to this challenge, register at
> [Action Recognition Codalab Challenge](https://competitions.codalab.org/competitions/25923)


This repo contains:

- Model definitions for TSN/TRN/TSM courtesy of their original authors.
- Training script
- Testing script
- Pretrained models on the EPIC-Kitchens 100 validation set

- For TBN see https://github.com/ekazakos/temporal-binding-network
- For SlowFast see https://github.com/epic-kitchens/epic-kitchens-slowfast

## Table of Contents
* [Environment setup](#environment-setup)
* [Prep data](#prep-data)
    * [RGB](#rgb)
    * [Optical Flow](#optical-flow)
    * [Validating the data](#validating-the-data)
* [Training](#training)
* [Testing](#testing)
    * [Evaluating models and competition submissions](#evaluating-models-and-competition-submissions)
* [Training from existing weights](#training-from-existing-weights)
* [Pretrained models](#pretrained-models)
* [Acknowledgements](#acknowledgements)
    * [TSN](#tsn)
    * [TRN](#trn)
    * [TSM](#tsm)
* [License](#license)


## Environment setup

We provide a conda environment definition in [`environment.yml`](./environment.yml
) that defines the dependencies you need to run this codebase. Simply set up the
 environment by running

```bash
$ conda env create -n epic-models -f environment.yml
$ conda activate epic-models
```

We then suggest you replace the installation of Pillow with Pillow-SIMD to improve
dataloading performance, however this is an optional step. (see
[fastai docs](https://fastai1.fast.ai/performance.html#faster-image-processing)
 for more details)

```bash
# The following steps are taken from
# https://docs.fast.ai/performance.html#installation

$ conda uninstall -y --force jpeg libtiff pillow
$ conda install -y -c conda-forge libjpeg-turbo gxx_linux-64
$ pip uninstall -y pillow
$ export CXX=x86_64-conda-linux-gnu-g++
$ export CC=x86_64-conda-linux-gnu-gcc
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd

# Make sure Pillow doesn't get reinstalled
$ conda install -c conda-forge jpeg libtiff
```

Check that Pillow-SIMD is installed, it should have a `.postX` suffix on the version
number:

```bash
$ python -c 'import PIL; print(PIL.__version__)'
7.0.0.post3
```

Check that Pillow-SIMD was built against libjpeg-turbo:

```bash
$ python -c 'import PIL.features; print(PIL.features.check_feature("libjpeg_turbo"))'
True
```

## Prep data

Gulp the train/validation/test sets from the provided extracted frames

### RGB
```bash
$ python src/gulp_data.py \
    /path/to/rgb/frames \
    gulp/rgb_train \
    /path/to/EPIC_100_train.pkl \
    rgb
$ python src/gulp_data.py \
    /path/to/rgb/frames \
    gulp/rgb_validation \
    /path/to/EPIC_100_validation.pkl \
    rgb
$ python src/gulp_data.py \
    /path/to/rgb/frames \
    gulp/rgb_test \
    /path/to/EPIC_100_test_timestamps.pkl
    rgb
```

### Optical Flow

First we need to convert the frame numbers from those for RGB frames to those of
the flow frames (since in 2018 we extracted optical flow for every other frame).

```bash
$ python src/convert_rgb_to_flow_frame_idxs.py \
    /path/to/EPIC_100_train.pkl \
    EPIC_100_train_flow.pkl
$ python src/convert_rgb_to_flow_frame_idxs.py \
    /path/to/EPIC_100_validation.pkl \
    EPIC_100_validation_flow.pkl
$ python src/convert_rgb_to_flow_frame_idxs.py \
    /path/to/EPIC_100_test_timestamps.pkl \
    EPIC_100_test_timestamps_flow.pkl
```

We can then proceed with gulping the data.
```bash

$ python src/gulp_data.py \
    /path/to/flow/frames \
    gulp/flow_train \
    EPIC_100_train_flow.pkl \
    flow
$ python src/gulp_data.py \
    /path/to/flow/frames \
    gulp/flow_validation \
    EPIC_100_validation_flow.pkl \
    flow
$ python src/gulp_data.py \
    /path/to/flow/frames \
    gulp/flow_test \
    EPIC_100_test_timestamps_flow.pkl \
    flow
```

### Validating the data

Check out `notebooks/dataset.ipynb` to visualise the gulped RGB and optical flow as
a sanity check.

## Training

We provide configurations for training the models to
reproduce the results in Table 3 of ["Rescaling Egocentric Vision"](https://arxiv.org/abs/2006.13256).

We first train networks on each modality separately, then we produce results on the
validation/test set and fuse the results of the modality pre-softmax by averaging them.
See the next section for how to do this.

Training is implemented using
[Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
 and configuration managed by [hydra](https://hydra.cc/docs/intro).

To train a network, run the following:

```bash
# See configs/tsn_rgb.yaml for an example configuration file.
# You can overwrite config files by passing key-value pairs as arguments
# You can change the config by setting --config-name to the name of a file in configs
# without the yaml suffix.
$ python src/train.py \
    --config-name tsn_rgb \
    data._root_gulp_dir=/path/to/gulp/root \
    data.worker_count=$(nproc) \
    learning.batch_size=64 \
    trainer.gpus=4 \
    hydra.run.dir=outputs/experiment-name

# View logs with tensorboard
$ tensorboard --logdir outputs/experiment-name --bind_all
```

If you want to resume a checkpoint partway through training, then run

```bash
$ python src/train.py \
    --config-name tsn_rgb \
    data._root_gulp_dir=/path/to/gulp/root \
    data.worker_count=$(nproc) \
    learning.batch_size=64 \
    trainer.gpus=4 \
    hydra.run.dir=outputs/experiment-name \
    +trainer.resume_from_checkpoint="'$PWD/outputs/experiment-name/lightning_logs/version_0/checkpoints/epoch=N.ckpt'"
```

Note the use of single quotes within the double quotes, this is to protect the string
from hydra interpreting the `=` as a malformed key-value pair.

Any keyword arguments can be injected into the pytorch-lightning
[`Trainer`](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api)
object through the CLI by using `+trainer.<kwarg>=<value>`.

## Testing

Once you have trained a model, you can test that model by using the `test.py` script
which takes the checkpoint file and writes a `prediction.pt` file containing the
model output for all examples in the validation or test set.

```bash
# Get model results on the validation set for computing metrics
$ python src/test.py \
    outputs/experiment-name/lightning_logs/version_0/checkpoints/epoch=N.ckpt \
    outputs/experiment-name/lightning_logs/version_0/results/val_results_epoch=N.pt \
    --split val

# Get model results on the test set for submission to the challenge
$ python src/test.py \
    outputs/experiment-name/lightning_logs/version_0/checkpoints/epoch=N.ckpt \
    outputs/experiment-name/lightning_logs/version_0/results/test_results_epoch=N.pt \
    --split test
```

You can fused results from multiple modalities:

```bash
$ python src/fuse.py \
    outputs/experiment-name-rgb/lightning_logs/version_0/results/test_results_epoch=N.pt \
    outputs/experiment-name-flow/lightning_logs/version_0/results/test_results_epoch=N.pt \
    experiment-name-fused.pt
```

These fused results can then be passed to the evaluation script or JSON submission
generation script like any other single-modality results file.


### Evaluating models and competition submissions

Please see details in https://github.com/epic-kitchens/C1-Action-Recognition for how
to evaluate the models in this repo.

## Training from existing weights

If you already have some weights you wish to use as an initialisation, you can use these
by specifying `+model.weights=path/to/weights` as an argument when training. This
must be a `torch.save` serialised state dictionary for the full model. If you only
have partial weights which you wish to use to initalise the model, then simply dump a
randomly initialise state dict for the model, and then inject the weights you have into that.

## Pretrained models

We provide models pretrained on the training set of EPIC-KITCHENS-100.

| Model | Modality | Action@1 (Val) | Action@1 (Test) | Link                                                         |
|-------|----------|----------------|-----------------|--------------------------------------------------------------|
| TSN   | RGB      | 27.40          | 24.11           | https://www.dropbox.com/s/4i99mzddk95edyq/tsn_rgb.ckpt?dl=1  |
| TSN   | Flow     | 22.86          | 24.62           | https://www.dropbox.com/s/res0i1ns7v30g9y/tsn_flow.ckpt?dl=1 |
| TRN   | RGB      | 32.64          | 29.54           | https://www.dropbox.com/s/l1cs7kozz3f03r4/trn_rgb.ckpt?dl=1  |
| TRN   | Flow     | 22.97          | 23.43           | https://www.dropbox.com/s/4rehj36vyip82mu/trn_flow.ckpt?dl=1 |
| TSM   | RGB      | 35.75          | 32.82           | https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1  |
| TSM   | Flow     | 27.79          | 27.99           | https://www.dropbox.com/s/8x9hh404k641rqj/tsm_flow.ckpt?dl=1 |

## Acknowledgements

If you make use of this repository, please cite our dataset papers:

```
@ARTICLE{Damen2020RESCALING,
   title={Rescaling Egocentric Vision},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {CoRR},
           volume    = {abs/2006.13256},
           year      = {2020},
           ee        = {http://arxiv.org/abs/2006.13256},
}

@INPROCEEDINGS{Damen2018EPICKITCHENS,
   title={Scaling Egocentric Vision: The EPIC-KITCHENS Dataset},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and Fidler, Sanja and
           Furnari, Antonino and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan
           and Perrett, Toby and Price, Will and Wray, Michael},
   booktitle={European Conference on Computer Vision (ECCV)},
   year={2018}
}
```

### TSN

We thank the authors of TSN for providing their
[codebase](https://github.com/yjxiong/tsn-pytorch), from which we took:

- The model definition
  ([`models.py`](https://github.com/yjxiong/tsn-pytorch/blob/master/models.py))
- The transforms
  ([`transforms.py`](https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py))


Please cite their work if you make use of this network

```
@InProceedings{wang2016_TemporalSegmentNetworks,
    title={Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
    author={Limin Wang and Yuanjun Xiong and Zhe Wang and Yu Qiao and Dahua Lin and
            Xiaoou Tang and Luc {Val Gool}},
    booktitle={The European Conference on Computer Vision (ECCV)},
    year={2016}
}
```

### TRN

We thank the authors of TRN for providing their
[codebase](https://github.com/zhoubolei/TRN-pytorch), from which we took:

- The model definition
  ([`models.py`](https://github.com/zhoubolei/TRN-pytorch/blob/master/models.py))
- The TRN module definition
  ([`TRNmodule.py`](https://github.com/zhoubolei/TRN-pytorch/blob/master/TRNmodule.py))

Please cite their work if you make use of this network

```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

### TSM

We thank the authors of TSM for providing their
[codebase](https://github.com/MIT-HAN-LAB/temporal-shift-module), from which we took:

- The model definition
  ([`ops/models.py`](https://github.com/mit-han-lab/temporal-shift-module/tree/master/ops/models.py))
- The TSM module definition
  ([`ops/temporal_shift.py`](https://github.com/mit-han-lab/temporal-shift-module/tree/master/ops/temporal_shift.py))

Please cite their work if you use this network

```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

## License

Copyright University of Bristol.
The repository is published under the [Creative Commons Attribution-NonCommercial 4.0
 International License](https://creativecommons.org/licenses/by-nc/4.0/). This means that you must
give appropriate credit, provide a link to the license, and indicate if changes were
made. You may do so in any reasonable manner, but not in any way that suggests the
licensor endorses you or your use. You may not use the material for commercial purposes.
