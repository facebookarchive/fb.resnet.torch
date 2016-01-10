--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local datasets = require 'datasets/init'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Model and criterion
local model, criterion = models.setup(opt)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local best1Err = math.huge
local best5Err = math.huge
for epoch = opt.epochNumber, opt.nEpochs do
   -- Train for a single epoch
   trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local top1Err, top5Err = trainer:test(epoch, valLoader)

   -- Save the model if it has the best top-1 error
   if top1Err < best1Err then
      print(' * Saving best model ', top1Err, top5Err)
      torch.save('model_best.t7', model)
      best1Err = top1Err
      best5Err = top5Err
   end
   torch.save('model_' .. epoch .. '.t7', model)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', best1Err, best5Err))
