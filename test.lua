--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
-- require 'torch'
require 'nn'

-- torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setnumthreads(1)

local M = {}
local Tester = torch.class('resnet.Tester', M)

function Tester:__init(model, opt)
   self.model = model
   self.nCrops = opt.tenCrop and 10 or 1
end

function Trainer:test(dataloader)
   self.model:evaluate()

   local predictions = {}
   local timer = torch.Timer()
   local size = dataloader:size()

   for n, sample in dataloader:run() do
      local output = model:forward(sample.input:cuda():contiguous()):float()
      if self.nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
          --:exp()
          :sum(2):squeeze(2)
       output = output / nCrops
    end

    for i=1,(#output)[1] do
       predictions[count] = {['name']=sample.path[i], ['output']=output[i]}
    end
    print((' | Test: [%d/%d]    Time %.3f '):format(
       n, size, timer:time().real))
    timer:reset()





    file = io.open('output.csv', 'w')

    for i=1,(len(predictions)-1) do
       print(predictions[i])
       s = predictions[i]['name']
       for j=1,4 do
            s = s..','..predictions[i]['output'][j]
       end
       file:write(s..'\n')
    end

    file.close()
end
