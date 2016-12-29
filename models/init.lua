--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):cuda()
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
      model.__memoryOptimized = nil
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local imageSize = opt.dataset == 'imagenet' and 224 or 32

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local sampleInput = torch.zeros(4,3,imageSize,imageSize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model)
   end

   -- Attach a spatial transformer as the first block of the network
   -- initialied as an identity affine transform
   if opt.attachSpatialTransformer then
     M.attachSpatialTransformer(model, imageSize)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   if(opt.printModel) then
     print("*************************************")
     print("********** Model Topology  **********")
     print("*************************************")
     print(model)
     print("*************************************")
     print("*************************************")
   end

   local criterion = nn.CrossEntropyCriterion():cuda()
   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

function M.attachSpatialTransformer(model, imageSize)
   local stn = require 'stn'
   local Convolution = cudnn.SpatialConvolution
   local ReLU = cudnn.ReLU
   local SBatchNorm = nn.SpatialBatchNormalization

   -- Initialization for ReLU networks described in "Delving Deep Into Rectifiers: Surprassing Human-Level Performance on ImageNet Classification"
	 function getRandomizedConvolutionalLayer(nInputPlane, nOutputPlane, stride, filterSize, padding)
      local convolution = Convolution(nInputPlane, nOutputPlane, filterSize, filterSize, stride, stride, padding, padding)
      local numberOfConnections = nInputPlane * nOutputPlane * filterSize * filterSize
      local standardDeviation = math.sqrt(2.0 / numberOfConnections)
      convolution.weight = torch.randn(convolution.weight:size()) * standardDeviation
      convolution.weight:cuda()
      convolution.bias:zero()
      return convolution
   end

   function getLocationNetwork(nOutputPlane, stride)
      local nInputPlane = 3 -- RGB raw image channels
      local spatialParameters = 4 -- Normal-Surface-Angle-Rotation + Scale + X-Translation + Y-Translation
      local localizationNetwork = nn.Sequential()
      localizationNetwork:add(getRandomizedConvolutionalLayer(nInputPlane, nOutputPlane, stride, 3, 1))
      localizationNetwork:add(SBatchNorm(nOutputPlane))
      localizationNetwork:add(ReLU(true))
      localizationNetwork:add(getRandomizedConvolutionalLayer(nOutputPlane, nOutputPlane, stride, 3, 1))
      localizationNetwork:add(SBatchNorm(nOutputPlane))
      localizationNetwork:add(ReLU(true))
      localizationNetwork:add(getRandomizedConvolutionalLayer(nOutputPlane, nOutputPlane, stride, 3, 1))
      localizationNetwork:add(SBatchNorm(nOutputPlane))
      localizationNetwork:add(ReLU(true))

      -- fully connected layer created using convolutional layer with filters that span the whole input spatial space
      local inputSizeInLastLayer = (imageSize / (stride * stride * stride) )
      local fullyConnectedLayerWithIdentityAffineTransform = Convolution(nOutputPlane, spatialParameters, inputSizeInLastLayer, inputSizeInLastLayer, 1, 1, 0, 0)
      fullyConnectedLayerWithIdentityAffineTransform.weight:zero()
      fullyConnectedLayerWithIdentityAffineTransform.bias[1] = 0.0 -- zero angle rotation
      fullyConnectedLayerWithIdentityAffineTransform.bias[2] = 1.0 -- no scaling
      fullyConnectedLayerWithIdentityAffineTransform.bias[3] = 0.0 -- zero x translation
      fullyConnectedLayerWithIdentityAffineTransform.bias[4] = 0.0 -- zero y translation

      localizationNetwork:add(fullyConnectedLayerWithIdentityAffineTransform)
      return localizationNetwork:cuda()
   end

   function getSpatialTransformer()
      local localization_network = getLocationNetwork(16, 2)
      local concatTable = nn.ConcatTable()
      local shortcutTranspose = nn.Transpose({3,4},{2,4})
      local spatialTransformerBranch = nn.Sequential()
      spatialTransformerBranch:add(localization_network)
      spatialTransformerBranch:add(stn.AffineTransformMatrixGenerator(true, true, true))
      spatialTransformerBranch:add(stn.AffineGridGeneratorBHWD(imageSize, imageSize))
      concatTable:add(shortcutTranspose)
      concatTable:add(spatialTransformerBranch)
      local spatialTransformerModule = nn.Sequential()
      spatialTransformerModule:add(concatTable)
      spatialTransformerModule:add(stn.BilinearSamplerBHWD())
      spatialTransformerModule:add(nn.Transpose({2,4},{3,4}))
      return spatialTransformerModule:cuda()
   end

   local spatialTransformer = getSpatialTransformer()
   model:insert(spatialTransformer, 1)
   return model
end

return M
