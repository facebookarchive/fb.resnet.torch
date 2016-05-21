--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel, opt)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   torch.save(paths.concat(opt.resume, modelFile), model)
   torch.save(paths.concat(opt.resume, optimFile), optimState)
   torch.save(paths.concat(opt.resume, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      torch.save(paths.concat(opt.resume, 'model_best.t7'), model)
   end
end

return checkpoint
