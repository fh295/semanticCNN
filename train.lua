
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'dataset'



local w2v
--[[
    NOTE!!!!
    FIND A WAY TO NOT HAVE TO CREATE A DUMMY DATALOADER JUST TO 
    BE ABLE TO GRAB THE W2V
--]]

if opt.crit == 'sem' or opt.crit == 'mse' or opt.crit == 'softsem' then
      dummy = dataLoader{
      paths = {paths.concat(opt.data, 'val')}, --train
      loadSize = {3, opt.imageSize, opt.imageSize}, --doesn't really matter
      sampleSize = {3, opt.cropSize, opt.cropSize},  -- doesn't really matter
      split = 100,
      verbose = true,
      wvectors = opt.wvectors,
   } 
  w2v = dummy:get_w2v()
end

if opt.crit == 'softsem' then
    local class_labels = torch.load('classes.t7')
    semantic_array = torch.Tensor(nClasses, opt.wvectors_dim)

    --fill up semantic array
    for i,c in ipairs(class_labels) do      
        local vector = w2v:getWordVector(c)
        semantic_array[i] = vector:clone()
    end
    nFixedParams = semantic_array:nElement()
    semantic_array = semantic_array:view(nFixedParams)
    semantic_array = semantic_array:cuda()
end



--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
	    local inputs, labels, vectors
	    vectors = torch.rand(1)
            if opt.crit == 'class' or opt.crit == 'softsem' then
            	inputs, labels = trainLoader:sample(opt.batchSize)
	    else
		inputs, vectors, labels = trainLoader:semanticsample(opt.batchSize, opt.neg_samples)
	    end 
	    return inputs, vectors, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local vectors = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()
if opt.crit == 'softsem' then
   nParameters = parameters:size()[1]
   parameters[{{nParameters-nFixedParams+1,nParameters}}] = semantic_array:clone()
end      


--semantic initialisation of 

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, vectorsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   if opt.crit == 'sem'  then
	vectors:resize(vectorsCPU:size()):copy(vectorsCPU)
        labels:resize(labelsCPU[1]:size()):copy(labelsCPU[1])
   elseif opt.crit == 'class' or opt.crit == 'softsem' then
   	labels:resize(labelsCPU:size()):copy(labelsCPU)
   else
	labels:resize(vectorsCPU:size()):copy(vectorsCPU)
   end

   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      -- format input data to be {images}
      local output = model:forward(inputs)

      --format input to criterion to be either {prediction} or {predictions, w_vectors}
      if opt.crit == 'class' or opt.crit == 'mse' or opt.crit == 'softsem' then
        outputs = output
      else
      	outputs = {output, vectors}
      end
      err = criterion:forward(outputs, labels)

      local grads = criterion:backward(outputs, labels)

      local gradOutputs 
      if opt.crit == 'class' or opt.crit == 'mse' or opt.crit == 'softsem' then 
	gradOutputs = grads   
      else
	gradOutputs = grads[1] -- cause we throw away the grads for the word embeddings
      end

      model:backward(inputs, gradOutputs)

      return err, gradParameters
      
   end

   optim.sgd(feval, parameters, optimState)
   if opt.crit == 'softsem' then
      parameters[{{nParameters-nFixedParams+1,nParameters}}] = semantic_array:clone()
   end      

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   local median = 0
   local sim = 0
   if opt.crit == 'class' or opt.crit == 'softsem' then
      do
          local _,prediction_sorted = outputs:float():sort(2, true) -- descending
          for i=1,opt.batchSize do
	      if prediction_sorted[i][1] == labelsCPU[i] then
	          top1_epoch = top1_epoch + 1;
	          top1 = top1 + 1
	      end
          end
          
      end
      top1 = top1 * 100 / opt.batchSize;
   elseif opt.crit == 'sem' then 
      top1, median, sim = w2v:eval_ranking(outputs[1]:float(), labelsCPU[1], labelsCPU[2],1, opt.neg_samples)
   else
     top1, median, sim = w2v:eval_ranking(outputs:float(), labelsCPU[1], labelsCPU[2],1, opt.neg_samples)
   end
      -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1 %.4f  (Sim %.4f Med %.4f) LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1, median, sim,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end
