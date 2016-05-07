-- reads the provided directory and extracts features in batch mode
-- the output file, features.t7 also has list of file names in the 
-- same order as those of the features
-- 
--  Usage: th extract-features-batch-mode.lua [MODEL] [DIRECTORY_CONTAINING_IMAGES] [BATCH_SIZE]

require 'torch'
require 'paths'


if #arg < 2 then
   io.stderr:write('Usage: th extract-features-batch-mode.lua [MODEL] [DIRECTORY_CONTAINING_IMAGES] [BATCH_SIZE]\n')
   os.exit(1)
end

require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'

-- Load the model
local model = torch.load(arg[1])
local batch_size = tonumber(arg[3])

-- lfs module helps to navigate directories
local lfs = require 'lfs'
local dir_path = arg[2]


-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

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


local file_all_list = {}

-- collect all the files, this helps to make the batches
for file in lfs.dir(dir_path) do
    if file~="." and file~=".." then
        table.insert(file_all_list, file)
    end
end


number_of_files = table.getn(file_all_list)
print("number of images to be processed " .. number_of_files )

local features = torch.FloatTensor(number_of_files, 2048)

for i=1,number_of_files,batch_size do
    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform 

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do 
        -- img_name = file_all_list:read()
        img_name = file_all_list[i+j-1] 

        if img_name  ~= nil then
            image_count = image_count + 1
            local img = image.load(dir_path..'/'..img_name, 3, 'float')
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

   -- in the extreme case where the last batch is a single file, do this instead
   if image_count == 1 then
       features[i]:copy(output)
   else
       features[{ {i, i-1+image_count}, {}  } ]:copy(output)
   end

end

torch.save('features.t7', {features=features, image_list=file_all_list})
print('saved features to '..arg[2]..'_features.t7')

