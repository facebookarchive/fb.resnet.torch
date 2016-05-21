--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local lmdb = require 'lmdb'

local M = {}
local ImagenetLMDBDataset = torch.class('resnet.ImagenetLMDBDataset', M)

function ImagenetLMDBDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   --self.dir = paths.concat(opt.data, split)
   --assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)

   self.env = lmdb.env{
       Path = paths.concat(opt.data, string.format('%s_lmdb', split)),
       Name = string.format('%s_lmdb', split)
   }
   assert(env:open(), 'directory does not exist: ' .. string.format('%s_lmdb', split))
   self.stat = env:stat() -- Current status

   self.reader = env:txn(true) --Read-only transaction
   self.idxs = torch.randperm(n_images)
   assert(self.imageInfo.imageClass:size(1) == #self.idxs, string.format('Something wrong with lmdb. The lmdb db should have %d number of items, but it has %d', self.imageInfo.imageClass:size(1), #self.idxs))
end

function ImagenetLMDBDataset:get(i)
   local item = reader:get(string.format("%07d", self.idxs[i]))
   return item
end

function ImagenetLMDBDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetLMDBDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetLMDBDataset
