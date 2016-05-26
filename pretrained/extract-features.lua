--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

-- USAGE
-- SINGLE FILE MODE
--          th extract-features.lua [MODEL] [FILE] ...
--
-- BATCH MODE
--          th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES]
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'

if #arg < 2 then
   io.stderr:write('Usage (Single file mode): th extract-features.lua [MODEL] [FILE] ... \n')
   io.stderr:write('Usage (Batch mode)      : th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES]  \n')
   os.exit(1)
end

-- get the list of files
local filenames = {}
local batchSize = 1

if not paths.filep(arg[1]) then
   io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
   os.exit(1)
end


if tonumber(arg[2]) ~= nil then -- batch mode ; collect file from directory
   local lfs  = require 'lfs'
   batchSize = tonumber(arg[2])
   local dir_path = arg[3]

   for file in lfs.dir(dir_path) do -- get the list of the files
      if file ~= '.' and file ~= '..' then
         table.insert(filenames, dir_path .. '/' .. file)
      end
   end

else -- single file mode ; collect file from command line
   for i=2, #arg do
      local f = arg[i]
      if not paths.filep(f) then
         io.stderr:write('file not found: ' .. f .. '\n')
         os.exit(1)
      else
         table.insert(filenames, f)
      end
   end
end

if batchSize > #filenames then
   batchSize = #filenames
end

-- Load the model
local model = torch.load(arg[1])

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

if #filenames == 1 then
   print('Extracting features for ' .. filenames[1])
else
   print('Extracting features for ' .. #filenames .. ' images')
end

local features

for i = 1, #filenames, batchSize do
   local sz = math.min(batchSize, #filenames - i + 1)
   local input = torch.FloatTensor(sz, 3, 224, 224)

   -- Load and preprocess the images for the batch
   for j = 1, sz do
      local filename = filenames[i+j-1]
      input[j] = transform(image.load(filename, 3, 'float'))
   end

   -- Get the output of the layer before the (removed) fully connected layer
   local output = model:forward(input:cuda())
   assert(output:dim() == 2)

   if not features then
      features = torch.FloatTensor(#filenames, output:size(2)):zero()
   end

   features[{ {i, i-1+sz}, {} }]:copy(output)
end

torch.save('features.t7', { features = features, image_list = filenames })
print('saved features to features.t7')
