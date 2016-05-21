--[[
  A script to conver images to lmdb dataset

  References
  1. https://github.com/facebook/fb.resnet.torch/blob/master/datasets/init.lua
  2. https://github.com/eladhoffer/ImageNet-Training/blob/master/CreateLMDBs.lua
]]--

local ffi = require 'ffi'
local image = require 'image'

-- Define local functions
local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

local function split(dataset)
  local tokens = {}
  for word in string.gmatch(dataset, '([^-]+)') do
      table.insert(tokens, word)
  end
  assert(tokens[2] == 'lmdb', string.format('opt.dataset should be <datset>-lmdb form; opt.dataset = %s', dataset))
  return tokens[1]
end

local function _loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

local function getItem(basedir, imageClass, imagePath, i)
   local path = ffi.string(imagePath[i]:data())
   local image = _loadImage(paths.concat(basedir, path))
   local class = imageClass[i]

   return {
      input = image,
      target = class,
   }
end



-- Init opt 
local opt = {}
opt.gen = 'gen'
opt.dataset = 'imagenet-lmdb'
opt.data = '/media/data1/image/ilsvrc15/ILSVRC2015/Data/CLS-LOC'

opt.data_lmdb = '/media/data1/image'
opt.shuffle = true
--print(opt)

-- Load imageInfo
local cachePath = paths.concat(opt.gen, split(opt.dataset) .. '.t7')
if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
   paths.mkdir('gen')

   local script = paths.dofile(split(opt.dataset) .. '-gen.lua')
   script.exec(opt, cachePath)
end
local imageInfo = torch.load(cachePath)
--print(imageInfo)

-- Create LMDB
local lmdb = require 'lmdb'

local train_env = lmdb.env{
    Path = paths.concat(opt.data_lmdb, 'train_lmdb'),
    Name = 'train_lmdb'
}

local val_env= lmdb.env{
    Path = paths.concat(opt.data_lmdb, 'val_lmdb'),
    Name = 'val_lmdb'
}

local path = ffi.string(imageInfo.train.imagePath[1]:data())
--local image = self:_loadImage(paths.concat(self.dir, path))
--local class = self.imageInfo.imageClass[i]
print(path)

local n_images = (#imageInfo.train.imagePath)[1]
print(string.format("n_image: %d", n_images))

local idxs
if opt.shuffle then
  idxs = torch.randperm(n_images)
else 
  idxs = torch.range(1, n_images)
end
print(string.format("opt.shuffle: %s, idxs[1]: %d", opt.shuffle, idxs[1]))

local basedir = paths.concat(imageInfo.basedir, 'train')
--local item = getItem(basedir, imageInfo.train.imageClass, imageInfo.train.imagePath, idxs[1])
--print(item.target)
----print(item.input)
--print(#item.input)
--print(item.input[1][1][1])

train_env:open()
local txn = train_env:txn()
local cursor = txn:cursor()
for i = 1, 1000 do --n_images do
    local item = getItem(basedir, imageInfo.train.imageClass, imageInfo.train.imagePath, idxs[i])

    cursor:put(string.format("%07d", i), item, lmdb.C.MDB_NODUPDATA)
    if i % 100 == 0 then
        txn:commit()
        print(train_env:stat())
        collectgarbage()
        txn = train_env:txn()
        cursor = txn:cursor()
    end
    xlua.progress(i, n_images)
end
txn:commit()
train_env:close()

--[[
local sys = require 'sys'
local n_test = 5000
sys.tic()
-------Read-------
train_env:open()
print(train_env:stat()) -- Current status
local reader = train_env:txn(true) --Read-only transaction
--local y = torch.Tensor(10,3,256,256)
local y = {}

local idxs = torch.randperm(n_test)
for i=1,n_test do
  local item = reader:get(string.format("%07d", idxs[i]))
  if i % 1000 == 0 then
    print(string.format('%d: %d', i, idxs[i]))
    print(#item.input)
  end 
  --print(item)
  --print(#item.input)
  --print(item.input[1][1][1])
end
reader:abort()
train_env:close()
print(sys.toc())
collectgarbage()

sys.tic()
-------Read-------
train_env:open()
print(train_env:stat()) -- Current status
local reader = train_env:txn(true) --Read-only transaction
--local y = torch.Tensor(10,3,256,256)
local y = {}

for i=1,n_test do
  local item = reader:get(string.format("%07d", i))
  if i % 1000 == 0 then
    print(string.format('%d: %d', i, i))
    print(#item.input)
  end
  --print(item)
  --print(#item.input)
  --print(item.input[1][1][1])
end
reader:abort()
train_env:close()
print(sys.toc())
]]--
