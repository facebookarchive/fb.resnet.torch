local generateGraph = require 'optnet.graphgen'

-- visual properties of the generated graph
-- follows graphviz attributes
local graphOpts = {
  displayProps =  {shape='ellipse',fontsize=14, style='solid'},
  nodeData = function(oldData, tensor)
    --return oldData .. '\n' .. 'Size: '.. tensor:numel()
    local text_sz = ''
    for i = 1,tensor:dim() do
      if i == 1 then 
        text_sz = text_sz .. '' .. tensor:size(i)
      else
        text_sz = text_sz .. ', ' .. tensor:size(i)
      end 
    end
    return oldData .. '\n' .. 'Size: {'.. text_sz .. '}\n' .. 'Mem size: ' .. tensor:numel()
  end
}

local function copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   local input = torch.CudaTensor()
   --print('type of input: ' .. torch.type(input))
   --print(sample:size())
   input:resize(sample:size()):copy(sample)

   return input
end

local function drawModel(model, input, name) 
  -- model: A network architecture
  -- input: The input for the given network architecture 
  -- name:  The model name (string).
  --        The files, '<name>.dot' and '<name>.svg' will be generated. 

  local input_ 
  if torch.type(input) == 'table' then
      input_ = {}
      --print('table: ', #input)
      for i = 1,#input do
        input_[i] = copyInputs(input[i]) 
        --print(torch.type(input_[i])) 
      end  
  else 
    input_ = copyInputs(input)
    --print(torch.type(input_))
  end

  g = generateGraph(model, input_, graphOpts)
  graph.dot(g, name, name)

  --print(torch.type(g))
  --print(g)
  --print(#g.nodes)
  --print(g.nodes[#g.nodes]:label())
  --print(g:leaves())

  return g 
end
 
---------------------------------------------------------------------------
-- Sample input 
local eps = 1e-12
local batch_size = 1
local n_rois = 10
local height = 299 
local width = 299 
 
local input_image = torch.floor(torch.rand(batch_size, 3, height, width) * 256)
local input_rois = torch.cat(torch.floor(torch.rand(n_rois, 1) * height/2), 
                             torch.floor(torch.rand(n_rois, 1) * width/2), 2)
input_rois = torch.cat(input_rois, 
                       torch.add(input_rois, 
                                 torch.cat(torch.floor(torch.rand(n_rois, 1) * height/2), 
                                           torch.floor(torch.rand(n_rois, 1) * width/2), 2)))
input_rois = torch.cat(torch.floor(torch.rand(n_rois,1) * batch_size) + 1, input_rois, 2)
--print(#input_rois)
--print(input_rois[1])
--print(#input_image)

---------------------------------------------------------------------------
-- Create net w/ options
print('Create network architectures')

--local createModel = paths.dofile('models/inceptionv4.lua')
local createModel = paths.dofile('models/inceptionv4aux.lua')

local opt = {}; 
opt.cudnn = 'fastest'

local model = createModel(opt) 

--local g = drawModel(model, input_image, 'inceptionv4-cls') 
local g = drawModel(model, input_image, 'inceptionv4-cls-aux') 

--local output = model:forward(input_image:cuda())
--print(output)

print('Done.')



