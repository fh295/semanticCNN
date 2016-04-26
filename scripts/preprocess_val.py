import os
import random
import shutil
import json
import pickle
SIZE = 50

concepts = []
#with open('../word2vec/rows.json') as f:
#	concepts = json.load(f)

with open('../word2vec/semantic_vector_dict.pkl') as f:
	concepts = pickle.load(f).keys()


mappings = {}
with open('../imagenet_labels/mapping.txt') as f:
	for line in f.readlines():
		line = line.strip()
		synset = line.split()[0]
		label = line.split()[1]
		if label in concepts:
			mappings[synset] = label

ids2synsets = {}
with open('../imagenet_labels/ILSVRC2012_mapping.txt') as f:
	for line in f.readlines():
		line = line.strip()
		ID = line.split()[0]
		synset = line.split()[1]
		ids2synsets[ID] = synset


byClass = {}
images2labels = {}
imagedir_in = "../../DATA/val"
base = "ILSVRC2012_val_"
i = 1
with open('../imagenet_labels/ILSVRC2012_validation_ground_truth.txt') as f:
		for line in f.readlines():
			ID = line.strip()
			instance = str(i).zfill(8)
			fileName = base+instance+".JPEG"
			synset = ids2synsets[ID]
			if synset in mappings:
				l = mappings[ids2synsets[ID]]
				if l not in byClass:
					byClass[l] = []
				byClass[l].append(fileName)
			i = i+1
		
print "read ",i," images"
print len(byClass)
#imagedir_out = "/home/angeliki/git/imagenet-multiGPU.torch/DATA/semantic_tmp/val"
imagedir_out = "../../DATA/val"
towrite = {}
for label in byClass:
	random.shuffle(byClass[label])
	towrite = byClass[label][:SIZE]
	newdir = os.path.join(imagedir_out,label)
	os.mkdir(newdir)
	for f in towrite:
		print os.path.join(imagedir_in,f), os.path.join(newdir,f)
		shutil.copyfile(os.path.join(imagedir_in,f),os.path.join(newdir,f))
	
	

		






