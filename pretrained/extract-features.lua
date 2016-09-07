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
local list_of_filenames = {}
local batch_size = 1

if not paths.filep(arg[1]) then
    io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
    os.exit(1)
end
    

if tonumber(arg[2]) ~= nil then -- batch mode ; collect file from directory
    
    local lfs  = require 'lfs'
    batch_size = tonumber(arg[2])
    dir_path   = arg[3]

    for file in lfs.dir(dir_path) do -- get the list of the files
        if file~="." and file~=".." then
            table.insert(list_of_filenames, dir_path..'/'..file)
        end
    end

else -- single file mode ; collect file from command line
    for i=2, #arg do
        f = arg[i]
        if not paths.filep(f) then
          io.stderr:write('file not found: ' .. f .. '\n')
          os.exit(1)
        else
           table.insert(list_of_filenames, f)
        end
    end
end

local number_of_files = #list_of_filenames

if batch_size > number_of_files then batch_size = number_of_files end

-- Load the model
local model = torch.load(arg[1]):cuda()

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

local features

for i=1,number_of_files,batch_size do
    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform 

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do 
        img_name = list_of_filenames[i+j-1] 

        if img_name  ~= nil then
            image_count = image_count + 1
            local img = image.load(img_name, 3, 'float')
            img = transform(img)
            img_batch[{j, {}, {}, {} }] = img
        end
    end

    -- if this is last batch it may not be the same size, so check that
    if image_count ~= batch_size then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

   -- Get the output of the layer before the (removed) fully connected layer
   local output = model:forward(img_batch:cuda()):squeeze(1)


   -- this is necesary because the model outputs different dimension based on size of input
   if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 

   if not features then
       features = torch.FloatTensor(number_of_files, output:size(2)):zero()
   end
       features[{ {i, i-1+image_count}, {}  } ]:copy(output)

end

torch.save('features.t7', {features=features, image_list=list_of_filenames})
print('saved features to features.t7')
