--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'hdf5'
require 'gnuplot'
require 'nn'
require 'LinearNB'


paths.dofile('dataset.lua')
paths.dofile('util.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Apply trained model to extract features from images from ALL layers')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-cache', 'imagenet/checkpoint', 'subdirectory containing cached files like mean, std etc')
cmd:option('-cropSize',224,'Crop size')
cmd:option('-imageSize',256,'Size of images')
cmd:option('-modelPath','','File to load the trained model')
cmd:option('-type_of_model','', 'Type of model considered (class or sem or softsem')
cmd:option('-imageList','/home/fh295/filespace2/DATA/stimuli/stimuliPaths.txt','A file containing all the images to extract features for')
cmd:option('-n_imgs',92,'How many images are in the imageList?') 

local opt = cmd:parse(arg or {})

cache = opt.cache 
cropSize = opt.cropSize
imageSize = opt.imageSize

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(cache, 'trainCache.t7')
local testCache = paths.concat(cache, 'testCache.t7')
local meanstdCache = paths.concat(cache, 'meanstdCache.t7')

local loadSize   = {3, imageSize, imageSize}
local sampleSize = {3, cropSize, cropSize}


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out
end

if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
end



print(string.format('Loading file %s',opt.modelPath))
local model = torch.load(opt.modelPath)
model:evaluate()
print(model)

-----------------------------      main starts here   ---------------------
-- read imageList
local fh,err = io.open(opt.imageList)
local n = opt.n_imgs
ii =1
-- line by line

all_images = torch.DoubleTensor(n, 3, cropSize, cropSize)
while true do
        imagePath = fh:read()
        if imagePath == nil then break end
        print(string.format('Extracting features for %s',imagePath))
	img = testHook({loadSize}, imagePath)
	if img:dim() == 3 then
  		img = img:view(1, img:size(1), img:size(2), img:size(3))
	end
	all_images[ii] = img
	ii = ii + 1
end
fh:close()

local saveFile = hdf5.open('../DATA/PREDICTIONS/'..opt.type_of_model.. '.h5','w')

---- save pixels
local pixels = all_images:view(n,all_images[1]:nElement())
saveFile:write('pixels',pixels)

all_images = all_images:cuda()
----------------------------------------------------------------------
-- extract/remove/save etc etc
local predictions = model:forward(all_images:cuda()):float()
saveFile:write('topLayer',predictions)
model:get(2):remove()
model:get(2):remove()
print(model)
predictions = model:forward(all_images:cuda()):float()
saveFile:write('bottomLayer',predictions)
model:get(2):remove()
model:get(2):remove()
model:get(2):remove()
print(model)
predictions = model:forward(all_images:cuda()):float()
saveFile:write('evenbottomLayer',predictions)


---------------------------------------

--[[
model:remove(2)
predictions = model:forward(all_images:cuda()):float()
predictions= predictions:view(n, predictions:size(2) *  predictions:size(3) *  predictions:size(4))
p_norm = predictions:norm(2,2)
predictions:cdiv(p_norm:expandAs(predictions))
sims = predictions * predictions:transpose(1,2)
gnuplot.imagesc(sims,'color')
--]]

saveFile:close()
