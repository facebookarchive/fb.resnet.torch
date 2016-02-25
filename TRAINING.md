Training recipes
----------------

### CIFAR-10

To train ResNet-20 on CIFAR-10 with 2 GPUs:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 20
```

To train ResNet-110 instead just change the `-depth` flag:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 110
```

To fit ResNet-1202 on two GPUs, you will need to use the [`-shareGradInput`](#sharegradinput) flag:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 1202 -shareGradInput true
```

### ImageNet

See the [installation instructions](INSTALL.md#download-the-imagenet-dataset) for ImageNet data setup.

To train ResNet-18 on ImageNet with 4 GPUs and 8 data loading threads:

```bash
th main.lua -depth 18 -nGPU 4 -nThreads 8 -batchSize 256 -data [imagenet-folder]
```

To train ResNet-34 instead just change the `-depth` flag:

```bash
th main.lua -depth 34 -nGPU 4 -nThreads 8 -batchSize 256 -data [imagenet-folder]
```
To train ResNet-50 on 4 GPUs, you will need to use the [`-shareGradInput`](#sharegradinput) flag:

```bash
th main.lua -depth 50 -nGPU 4 -nThreads 8 -batchSize 256 -shareGradInput true -data [imagenet-folder]
```

To train ResNet-101 or ResNet-152 with batch size 256, you may need 8 GPUs:

```bash
th main.lua -depth 152 -nGPU 8 -nThreads 12 -batchSize 256 -shareGradInput true -data [imagenet-folder]
```

## Useful flags

For a complete list of flags, run `th main.lua --help`.

### shareGradInput

The `-shareGradInput` flag enables sharing of `gradInput` tensors between modules of the same type. This reduces
memory usage. It works correctly with the included ResNet models, but may not work for other network architectures. See 
[models/init.lua](models/init.lua#L42-L60) for the implementation.

The `shareGradInput` implementation may not work with older versions of the `nn` package. Update your `nn` package by running `luarocks install nn`.

### shortcutType

The `-shortcutType` flag selects the type of shortcut connection. The [ResNet paper](http://arxiv.org/abs/1512.03385) describes three different shortcut types:
- `A`: identity shortcut with zero-padding for increasing dimensions. This is used for all CIFAR-10 experiments.
- `B`: identity shortcut with 1x1 convolutions for increasing dimesions. This is used for most ImageNet experiments.
- `C`: 1x1 convolutions for all shortcut connections.
