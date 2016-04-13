--TODO: FIX THE EVALUATION CODE


--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center, loss
local timer = torch.Timer()

if opt.crit == 'sem' or 'mse' then
      dummy = dataLoader{
      paths = {paths.concat(opt.data, 'test-small')}, --train
      loadSize = {3, opt.imageSize, opt.imageSize}, --doesn't really matter
      sampleSize = {3, opt.cropSize, opt.cropSize},  -- doesn't really matter
      split = 100,
      verbose = true,
      wvectors = opt.wvectors,
   }
  w2v = dummy:get_w2v()
end


function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels
            if opt.crit == 'class' or opt.crit == 'mse' or opt.crit == 'softsem' then 
		    inputs, labels = testLoader:get(indexStart, indexEnd)
	    else
		   inputs, labels = testLoader:getSemantic(indexStart, indexEnd)
            end
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, top1_center))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   if opt.crit == 'sem' then
	   labels:resize(labelsCPU[2]:size()):copy(labelsCPU[2])
   else
	   labels:resize(labelsCPU:size()):copy(labelsCPU)
   end

   local outputs = model:forward(inputs)
   cutorch.synchronize()
   local pred = outputs:float()

   local median = 0
   local sim = 0
   local top1 = 0
   if opt.crit == 'class' or opt.crit == 'softsem' then
   	local _, pred_sorted = pred:sort(2, true)
   	for i=1,pred:size(1) do
      		local g = labelsCPU[i]
      		if pred_sorted[i][1] == g then top1 = top1 + 1 end
   	end
	top1_center = top1 + top1_center
   elseif opt.crit == 'sem' then
      top1, sim, median = w2v:eval_ranking(pred, labelsCPU[1], labelsCPU[2],1, opt.neg_samples)
      top1 = top1*opt.batchSize/100
      top1_center = top1_center + top1
   end
   if batchNumber % opt.batchSize == 0 then
      print(('Epoch: Testing [%d][%d/%d] -- (top1: %d, median: %d)'):format(epoch, batchNumber, nTest, top1, median))
   end
end
