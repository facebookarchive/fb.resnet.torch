Trained ResNet Torch models
============================

These are ResNet models trainined on ImageNet. The accuracy on the ImageNet validation set are included below.

- [ResNet-18](https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7)
- [ResNet-34](https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7)
- [ResNet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7)
- [ResNet-101](https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7)
- [ResNet-152](https://d2j0dndfm35trm.cloudfront.net/resnet-152.t7)
- [ResNet-200](https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7)

The ResNet-50 model has a batch normalization layer after the addition, instead of immediately after the convolution layer. The ResNet-200 model is the full pre-activation variant from ["Identity Mappings in Deep Residual Networks"](http://arxiv.org/abs/1603.05027).

##### ImageNet 1-crop error rates (224x224)

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 30.43       | 10.76       |
| ResNet-34     | 26.73       | 8.74        |
| ResNet-50     | 24.01       | 7.02        |
| ResNet-101    | 22.44       | 6.21        |
| ResNet-152    | 22.16       | 6.16        |
| ResNet-200    | 21.66 <sup>[1](#notes)</sup> | 5.79        |

##### ImageNet 10-crop error rates

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 28.22       | 9.42        |
| ResNet-34     | 24.76       | 7.35        |
| ResNet-50     | 22.24       | 6.08        |
| ResNet-101    | 21.08       | 5.35        |
| ResNet-152    | 20.69       | 5.21        |
| ResNet-200    | 20.15       | 4.93        |

##### ImageNet charts

See the [convergence plots](CONVERGENCE.md) for charts of training and validation error and training loss after every epoch.

### Fine-tuning on a custom dataset

Your images don't need to be pre-processed or packaged in a database, but you need to arrange them so that your dataset contains a `train` and a `val` directory, which each contain sub-directories for every label. For example:

```
train/<label1>/<image.jpg>
train/<label2>/<image.jpg>
val/<label1>/<image.jpg>
val/<label2>/<image.jpg>
```

You can then use the included [ImageNet data loader](../datasets/imagenet.lua) with your dataset and train with the `-resetClassifer` and `-nClasses` options:

```bash
th main.lua -retrain resnet-50.t7 -data [path-to-directory-with-train-and-val] -resetClassifier true -nClasses 80
```

The labels will be sorted alphabetically. The first output of the network corresponds to the label that comes first alphabetically.

You can find how to create custom data loader in [datasets](../datasets) readme.

### Classification
To get the top 5 predicted of a model for a given input image, you can use the [`classify.lua`](classify.lua) script. For example:
```bash
th classify.lua resnet-101.t7 img1.jpg img2.jpg ...
``` 
Example output:
```
Classes for     cat.jpg
0.77302575111389        Egyptian cat
0.060410376638174       tabby, tabby cat 
0.040622022002935       tiger cat
0.025837801396847       lynx, catamount
0.018691379576921       window screen
```


### Extracting image features

The [`extract-features.lua`](extract-features.lua) script will extract the image features from an image and save them as a serialized Torch tensor. To use it, first download one of the trained models above. Next run it using

```bash
th extract-features.lua resnet-101.t7 img1.jpg img2.jpg ...
```

This will save a file called `features.t7` in the current directory. You can then load the image features in Torch.

```lua
local features = torch.load('features.t7')
```

### Notes
<sup>1</sup> This is on a test crop of size 224x224. On a test crop of size 320x320, the error rate is 20.1/4.8.
