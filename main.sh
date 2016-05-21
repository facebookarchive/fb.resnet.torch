#!/bin/bash

filename=$(date +"%Y:%m:%d-%H:%M:%S:%N")".log"

#th main.lua -depth 50 -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/

th main.lua -netType inceptionv4 -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/inceptionv4/ 2>&1 | tee /media/data0/cache/log/$filename

#th main.lua -netType inceptionv4aux -batchSize 64 -nGPU 2 -nThreads 16 -shareGradInput true -data /media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC/ -resume /media/data0/cache/inceptionv4/
