--[[
from here: https://github.com/rotmanmi/word2vec.torch
--]]
require 'paths'
require 'hdf5'
local cjson = require 'cjson'

torch.setdefaulttensortype('torch.FloatTensor')

local Word2vec = torch.class('Word2vec')

function Word2vec:__init(dir, classes)

	self.classes = classes
	self.w2v = {}
	local f = dir .. '/vectors.h5'
	if not paths.filep(f .. '.t7') then
		print(string.format('Ready to load word vectors from %s', f))
	        self.w2v = self:_bintot7(dir)
	else
		print(string.format('Ready to load word vectors from %s', f .. '.t7'))
        	self.w2v = torch.load(dir ..'/vectors.h5'.. '.t7')
        end
        print('Done reading word2vec data.')
end

function Word2vec:_read_json(path)
        local file = io.open(path, 'r')
        local text = file:read()
        file:close()
        local info = cjson.decode(text)
        return info
end


function Word2vec:_bintot7(dir)

       
	local f = dir .. '/vectors.h5'
	local h5_file = hdf5.open(f, 'r')

	local vecs_size = h5_file:read('/vectors'):dataspaceSize()
	local words = vecs_size[1]
	local dims = vecs_size[2]

	local M = h5_file:read('/vectors'):partial({1,words},{1,dims})

	local rows = self:_read_json(dir .. '/rows.json')
        --Writing Files
        word2vec = {}
        word2vec.M = torch.FloatTensor(M:size())
	word2vec.w2vvocab = {}
	word2vec.v2wvocab = {}
	for i=1,words do
		local w = rows[i]
		word2vec.v2wvocab[i] = w
		word2vec.w2vvocab[w] = i
		--normalize to unit norm	
                local n = M[i]:norm()
 		word2vec.M[i] = M[i]/n
	end
        
	torch.save(f .. '.t7',word2vec)
        print('Writing t7 File for future usage.')

        return word2vec
end



function Word2vec:getVector(label)
	local ind  = self.w2v.w2vvocab[self.classes[label]]
	return self.w2v.M[ind]
       
end

function Word2vec:getWordVector(label)
        local ind  = self.w2v.w2vvocab[label]
        return self.w2v.M[ind]

end

function Word2vec:eval_ranking(predictions, labels, classes, k, neg_samples)

  local els = predictions:size(1)
 
  -- normalize to have multiplication be cosine
  local p_norm = predictions:norm(2,2)
  predictions:cdiv(p_norm:expandAs(predictions))

  -- cosine
  local cosine = predictions * self.w2v.M:transpose(1,2)

  local sim = 0
  -- ranking
  local ranking = torch.Tensor(els/(neg_samples+1))
  local topk = 0
  local tot = 0

  for s = 1,els do
    
   if labels[s] == 1 then
    	local _,index = torch.sort(cosine:select(1,s),true) -- sort rows

    	local ind  = self.w2v.w2vvocab[self.classes[classes[s]]]
    	local not_found = true
    	local r = 1
    	while not_found do
      		if index[r]==ind then
        		ranking[tot+1] = r
        		not_found = false
        		if r <= k then topk = topk+1 end
      		end
      		r = r + 1
    	end
        tot = tot + 1 
        sim = sim + cosine[s][s]
        --print(string.format('Gold: %s -- Predicted %s ',self.classes[classes[s]], self.w2v.v2wvocab[index[1]])) 
   end
  end

  median = torch.median(ranking)[1]
  
  return topk * 100 /tot , sim/tot, median
end

