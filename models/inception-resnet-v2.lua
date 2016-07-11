
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function addConv(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw)
  assert(not (nInputPlane == nil)) 
  assert(not (nOutputPlane == nil))
  assert(not (kh == nil)) 
  assert(not (kw == nil)) 
  local sh = sh or 1
  local sw = sw or 1
  local ph = ph or 0
  local pw = pw or 0

  local layer = nn.Sequential():add(Convolution(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw))
                               :add(SBatchNorm(nOutputPlane))
                               :add(ReLU(true))
  return layer
end 

local function addConvLinear(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw) -- addConv without ReLU
  assert(not (nInputPlane == nil))
  assert(not (nOutputPlane == nil))
  assert(not (kh == nil))
  assert(not (kw == nil))
  local sh = sh or 1
  local sw = sw or 1
  local ph = ph or 0
  local pw = pw or 0

  local layer = nn.Sequential():add(Convolution(nInputPlane, nOutputPlane, kh, kw, sh, sw, ph, pw))
                               :add(SBatchNorm(nOutputPlane))
  return layer
end

local function Stem(fs_start) -- fig 3 
  local fs0 = {32, 32, 64}; fs0[0] = fs_start
  local net = nn.Sequential()
  ------------------- nInputPlane, nOutputPlane, k, k, s, s, p, p
  net:add(addConv(fs0[0], fs0[1], 3, 3, 2, 2, 0, 0))
  net:add(addConv(fs0[1], fs0[2], 3, 3, 1, 1, 0, 0))
  net:add(addConv(fs0[2], fs0[3], 3, 3, 1, 1, 1, 1))

  local fs1a = {};   fs1a[0] = fs0[#fs0]
  local fs1b = {96}; fs1b[0] = fs0[#fs0]
  local concat1 = nn.ConcatTable()
  concat1:add(Max(3, 3, 2, 2, 0, 0))
  concat1:add(addConv(fs1b[0], fs1b[1], 3, 3, 2, 2, 0, 0))

  net:add(concat1)
  net:add(nn.JoinTable(2, 4))

  local fs2a = {64, 96};         fs2a[0] = fs1a[#fs1a] + fs1b[#fs1b]
  local fs2b = {64, 64, 64, 96}; fs2b[0] = fs1a[#fs1a] + fs1b[#fs1b]
  local concat2 = nn.ConcatTable()
  concat2:add(nn.Sequential():add(addConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
                             :add(addConv(fs2a[1], fs2a[2], 3, 3, 1, 1, 0, 0))
                             )
  concat2:add(nn.Sequential():add(addConv(fs2b[0], fs2b[1], 1, 1, 1, 1, 0, 0))
                             :add(addConv(fs2b[1], fs2b[2], 7, 1, 1, 1, 3, 0))
                             :add(addConv(fs2b[2], fs2b[3], 1, 7, 1, 1, 0, 3))
                             :add(addConv(fs2b[3], fs2b[4], 3, 3, 1, 1, 0, 0))
                             )
  net:add(concat2)
  net:add(nn.JoinTable(2, 4))

  local fs3a = {192}; fs3a[0] = fs2a[#fs2a] + fs2b[#fs2b]
  local fs3b = {};    fs3b[0] = fs2a[#fs2a] + fs2b[#fs2b]
  local concat3 = nn.ConcatTable()
  concat3:add(addConv(fs3a[0], fs3a[1], 3, 3, 2, 2, 0, 0))
         :add(Max(3, 3, 2, 2, 0, 0)) 
  net:add(concat3)
  net:add(nn.JoinTable(2, 4))

  local fs_final = fs3a[#fs3a] + fs3b[#fs3b]
  return net, fs_final
end

local function InceptionResnetA(fs_start) -- fig 16
  local path1 = nn.Identity()

  local fs2a = {32};         fs2a[0] = fs_start 
  local fs2b = {32, 32};     fs2b[0] = fs_start
  local fs2c = {32, 48, 64}; fs2c[0] = fs_start
  local fs2  = {fs_start};  --local fs2  = {384};
                             fs2[0]  = fs2a[#fs2a] + fs2b[#fs2b] + fs2c[#fs2c] 
  local path2a = nn.Sequential():add(addConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
  local path2b = nn.Sequential():add(addConv(fs2b[0], fs2b[1], 1, 1, 1, 1, 0, 0))
                                :add(addConv(fs2b[1], fs2b[2], 3, 3, 1, 1, 1, 1))
  local path2c = nn.Sequential():add(addConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 0, 0))
                                :add(addConv(fs2c[1], fs2c[2], 3, 3, 1, 1, 1, 1))
                                :add(addConv(fs2c[2], fs2c[3], 3, 3, 1, 1, 1, 1))
  local path2  = nn.Sequential():add(nn.ConcatTable():add(path2a)
                                                     :add(path2b)
                                                     :add(path2c)
                                                     )
                                :add(nn.JoinTable(2, 4))
                                :add(addConvLinear(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                :add(nn.MulConstant(0.1))

  local net = nn.Sequential()
  net:add(nn.ConcatTable():add(path1)
                          :add(path2)
                          )
  net:add(nn.CAddTable(true)) 
  net:add(ReLU(true))
 
  local fs_final = fs2[#fs2] 
  assert(fs_final == fs_start)
 
  return net, fs_final
end

local function ReductionA(fs_start, k, l, m, n) -- fig 7
  local net = nn.Sequential()
  
  local concat = nn.ConcatTable()
  concat:add(Max(3, 3, 2, 2, 0, 0))                  -- path1
  concat:add(addConv(fs_start, n, 3, 3, 2, 2, 0, 0)) -- path2
  concat:add(nn.Sequential():add(addConv(fs_start, k, 1, 1, 1, 1, 0, 0)) -- path3
                            :add(addConv(k, l, 3, 3, 1, 1, 1, 1))
                            :add(addConv(l, m, 3, 3, 2, 2, 0, 0)))
  net:add(concat)
  net:add(nn.JoinTable(2, 4))

  local fs_final = fs_start + n + m 

  return net, fs_final 
end

local function InceptionResnetB(fs_start) -- fig 17
  local path1 = nn.Identity()

  local fs2a = {192};           fs2a[0] = fs_start
  local fs2c = {128, 160, 192}; fs2c[0] = fs_start
  local fs2  = {fs_start}; --local fs2  = {1152};
                                fs2[0]  = fs2a[#fs2a] + fs2c[#fs2c]
  local path2a = nn.Sequential():add(addConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
  local path2c = nn.Sequential():add(addConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 0, 0))
                                :add(addConv(fs2c[1], fs2c[2], 1, 7, 1, 1, 0, 3))
                                :add(addConv(fs2c[2], fs2c[3], 7, 1, 1, 1, 3, 0))
  local path2  = nn.Sequential():add(nn.ConcatTable():add(path2a)
                                                     :add(path2c)
                                                     )
                                :add(nn.JoinTable(2, 4))
                                :add(addConvLinear(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                :add(nn.MulConstant(0.1))

  local net = nn.Sequential()
  net:add(nn.ConcatTable():add(path1)
                          :add(path2)
                          )
  net:add(nn.CAddTable(true))
  net:add(ReLU(true))

  local fs_final = fs2[#fs2]
  assert(fs_final == fs_start)

  return net, fs_final
end

local function ReductionB(fs_start) -- fig 18
  local path1 = Max(3, 3, 2, 2, 0, 0)
  local fs2 = {256, 384}; fs2[0] = fs_start 
  local path2 = nn.Sequential():add(addConv(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                               :add(addConv(fs2[1], fs2[2], 3, 3, 2, 2, 0, 0))
  --local fs3 = {256, 288}; fs3[0] = fs_start -- in the paper, but seems to have typo
  local fs3 = {256, 256}; fs3[0] = fs_start 
  local path3 = nn.Sequential():add(addConv(fs3[0], fs3[1], 1, 1, 1, 1, 0, 0))
                               :add(addConv(fs3[1], fs3[2], 3, 3, 2, 2, 0, 0))
  --local fs4 = {256, 288, 320}; fs4[0] = fs_start -- in the paper, but seems to have typo
  local fs4 = {256, 256, 256}; fs4[0] = fs_start
  local path4 = nn.Sequential():add(addConv(fs4[0], fs4[1], 1, 1, 1, 1, 0, 0))
                               :add(addConv(fs4[1], fs4[2], 3, 3, 1, 1, 1, 1))
                               :add(addConv(fs4[2], fs4[3], 3, 3, 2, 2, 0, 0))
                               
  local concat = nn.ConcatTable()
  concat:add(path1)
  concat:add(path2)
  concat:add(path3)
  concat:add(path4)
 
  local net = nn.Sequential()
  net:add(concat)
  net:add(nn.JoinTable(2,4))

  local fs_final = fs_start + fs2[#fs2] + fs3[#fs3] + fs4[#fs4]
  
  return net, fs_final 
end

local function InceptionResnetC(fs_start) -- fig 19
  local path1 = nn.Identity()

  local fs2a = {192};           fs2a[0] = fs_start
  local fs2c = {192, 224, 256}; fs2c[0] = fs_start
  local fs2  = {fs_start}; --local fs2  = {2048};
                                fs2[0]  = fs2a[#fs2a] + fs2c[#fs2c]
  local path2a = nn.Sequential():add(addConv(fs2a[0], fs2a[1], 1, 1, 1, 1, 0, 0))
  local path2c = nn.Sequential():add(addConv(fs2c[0], fs2c[1], 1, 1, 1, 1, 0, 0))
                                :add(addConv(fs2c[1], fs2c[2], 1, 3, 1, 1, 0, 1))
                                :add(addConv(fs2c[2], fs2c[3], 3, 1, 1, 1, 1, 0))
  local path2  = nn.Sequential():add(nn.ConcatTable():add(path2a)
                                                     :add(path2c)
                                                     )
                                :add(nn.JoinTable(2, 4))
                                :add(addConvLinear(fs2[0], fs2[1], 1, 1, 1, 1, 0, 0))
                                :add(nn.MulConstant(0.1))

  local net = nn.Sequential()
  net:add(nn.ConcatTable():add(path1)
                          :add(path2)
                          )
  net:add(nn.CAddTable(true))
  net:add(ReLU(true))

  local fs_final = fs2[#fs2]
  assert(fs_final == fs_start)

  return net, fs_final
end



local function createModel(opt)
   local model = nn.Sequential()

   -- Add Stem module
   local stem, fs = Stem(3)
   model:add(stem)

   -- Add Inception-resnet-A modules (x5)
   for i = 1, 5 do 
     local irA, fs  = InceptionResnetA(fs)
     model:add(irA)
   end

   -- Add Reduction-A module 
   local rA, fs = ReductionA(fs, 256, 256, 384, 384)
   model:add(rA)

   -- Add Inception-resnet-B modules (x10)
   for i = 1, 10 do 
     local irB, fs = InceptionResnetB(fs)
     model:add(irB)
   end

   -- Add Reduction-B module
   local rB, fs = ReductionB(fs)
   model:add(rB)

   -- Add Inception-resnet-C modules (x5)
   for i = 1, 5 do
     local irC, fs = InceptionResnetC(fs)
     model:add(irC)
   end

   local nFeatures = fs -- set final channels

   -- Add Average Pooling
   model:add(Avg(8, 8, 1, 1)) 
   model:add(nn.View(nFeatures):setNumInputDims(3))

   -- Add Dropout (keep 0.8)
   model:add(nn.Dropout(0.2))
 
   -- Add Classifier 
   model:add(nn.Linear(nFeatures, 1000))

   -- Init
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   -- Convert to cuda 
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
