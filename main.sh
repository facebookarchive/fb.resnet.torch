#!/bin/bash

filename=$(date +"%Y:%m:%d-%H:%M:%S:%N")".log"

#th main.lua -depth 50 -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/

th main.lua -netType inception-resnet-v2 -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/inception-resnet-v2/ 2>&1 | tee /media/data0/cache/log/$filename

#th main.lua -netType inception-resnet-v2-aux -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/inception-resnet-v2/
