#converts a pkl file to an h5 file

import numpy as np
import json
import h5py
import pickle


#load vectors
vec_file = "semantic_vector_dict.pkl"
with open(vec_file,"r") as f:
	vecs = pickle.load(f)

#write keys
rows = vecs.keys()
keys_file = "rows.json"
with open(keys_file,"w") as f:
	json.dump(rows, f)

a = np.zeros([len(rows),vecs[rows[0]].shape[0]])
for i,k in enumerate(rows):
	a[i] = vecs[k]
 
#write vectors
vec_file = "vectors.h5"
f = h5py.File(vec_file, "w")
dset = f.create_dataset("vectors", a.shape, dtype='f', data=a)



