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

function checkpoint.save(epoch, model, optimState, bestModel, folder)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   if not paths.dirp(path) then
      paths.mkdir(path)
   end

   local modelFile = paths.concat(folder, 'model_' .. epoch .. '.t7')
   local optimFile = paths.concat(folder, 'optimState_' .. epoch .. '.t7')
   local latestFile = paths.concat(folder, 'latest.t7')

   torch.save(modelFile, model)
   torch.save(optimFile, optimState)
   torch.save(latestFile, {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      local bestFile = paths.concat(folder, 'model_best.t7')
      torch.save(bestFile, model)
   end
end

return checkpoint
