# U-Net: Semantic segmentation with PyTorch

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)

Originally adapted from
https://github.com/milesial/Pytorch-UNet
Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

### Preparation

Starter code
```console
import os
data_path = "G:/data/vision/numpy_images_for_classification-selected"
output_path = os.path.join(os.getcwd(), "Pytorch-UNet/data")
val = "Nkx_WT_298_Week12_analysis"

files = os.listdir(data_path)
print(files[0:])

#split files to load in pytorch and overfit on
for f in files:
    f_name = os.path.splitext(f)[0]
    f_npy = np.load(os.path.join(data_path, files[0]))

    f_npy_split = f_npy.reshape(-1, *f_npy.shape[2:])
    #print(f_npy_split.shape)
    for i in range(f_npy_split.shape[0]):
        if f_name == val:
            path = os.path.join(output_path, "val", f_name) + "_{}.npy".format(i)
        else:
            path = os.path.join(output_path, "imgs", f_name) + "_{}.npy".format(i)
        #print(path)
        np.save(path,  f_npy_split[i])
```

### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
