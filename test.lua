--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
-- require 'torch'
--
-- torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setnumthreads(1)

local M = {}
local Tester = torch.class('resnet.Tester', M)

function Tester:__init(model, opt)
   self.model = model
   self.nCrops = opt.tenCrop and 10 or 1
end

function Tester:test(dataloader)
   self.model:evaluate()

   local timer = torch.Timer()
   local size = dataloader:size()
   local numImages = dataloader.__size
   local indices = torch.Tensor(numImages):zero()
   local numClasses = #dataloader.dataset.classList
   local predictions = torch.Tensor(numImages, numClasses):zero()
   local numProcessed = 0

   for n, sample in dataloader:run() do
      local output = self.model:forward(sample.input:cuda():contiguous()):float()
      if self.nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / self.nCrops, self.nCrops, output:size(2))
          --:exp()
          :sum(2):squeeze(2)
         output = output / self.nCrops
      end

      local start = numProcessed + 1
      local stop = numProcessed + output:size(1)
      indices[{{start, stop}}] = sample.idx
      predictions[{{start, stop}, {}}] = output
      print((' | Test: [%d/%d]    Time %.3f '):format(
             n, size, timer:time().real))
      timer:reset()
      numProcessed = numProcessed + output:size(1)
   end





    file = io.open('output.csv', 'w')

    for i=1,(predictions:size(1)) do
       print(predictions[i])
       s = indices[i]
       for j=1,4 do
            s = s..','..predictions[i][j]
       end
       file:write(s..'\n')
    end

    file.close()
end

return M.Tester
