# takes learned representations and computes the RDM
# representations stored in h5 format

import pickle
import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.spatial.distance import pdist#
import h5py

model = sys.argv[1]
sim = sys.argv[2]
layer = sys.argv[3]

ALL = []

if model == 'words' or model=='ALL':

	word_vecs_file = "/home/fh295/Documents/Deep_learning_Bengio/arctic/defgen/D_medium_cbow_pdw_8B.pkl"
	with open(word_vecs_file,'r') as f:
		word_vecs = pickle.load(f)

	stimuli_words_file = "/home/fh295/filespace2/DATA/stimuli/mapping.txt"
	with open(stimuli_words_file,'r') as f:
		stimuli_words = [w.strip() for w in f.readlines()]
	
	vecs  = []
	rows = []
	for w in stimuli_words:
		if w in word_vecs:
			vecs.append(word_vecs[w])
			rows.append(w)
		else:
			print w," not in dict"

	#compute all pairwise similarities
	if sim == 'corr':
		m = 1-np.corrcoef(vecs)
		ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
	else:
		m =  pdist(vecs, 'cosine')
		ALL.append(m)

if model == 'humans' or model == 'ALL':
	m = np.loadtxt("/home/fh295/filespace2/DATA/stimuli/RDM_hIT_fig1.txt")
	ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])

if model == "pixels" or model == "ALL":
	f = h5py.File('/home/fh295/filespace2/DATA/PREDICTIONS/class.h5','r')
	vecs = f["pixels"]
	#compute all pairwise similarities
        if sim == 'corr':
                m = 1-np.corrcoef(vecs)
                ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
        else:
                m = pdist(vecs, 'cosine')
                ALL.append(m)
	f.close()	


if model == "sem" or model == "ALL":
	f = h5py.File('/home/fh295/filespace2/DATA/PREDICTIONS/sem.h5','r')
        vecs = f[layer]
	#compute all pairwise similarities
        if sim == 'corr':
                m = 1-np.corrcoef(vecs)
                ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
                print m[0][0],m[0][1],m[0][2]
        else:
                m = pdist(vecs, 'cosine')
                ALL.append(m)
	f.close()

if model == "class" or model == "ALL":
        f = h5py.File('/home/fh295/filespace2/DATA/PREDICTIONS/class.h5','r')
        vecs = f[layer]
	#compute all pairwise similarities
        if sim == 'corr':
                m = 1-np.corrcoef(vecs)
                ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
        else:
                m = pdist(vecs, 'cosine')
                ALL.append(m)
	f.close()

if model == "softsem" or model == "ALL":
        f = h5py.File('/home/fh295/filespace2/DATA/PREDICTIONS/softsem.h5','r')
        vecs = f[layer]
	#compute all pairwise similarities
        if sim == 'corr':
                m = 1-np.corrcoef(vecs)
                ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
        else:
                m = pdist(vecs, 'cosine')
                ALL.append(m)
	f.close()

if model == "vgg" or model == "ALL":
        f = h5py.File('/home/fh295/filespace2/DATA/PREDICTIONS/vgg.h5','r')
        vecs = np.array(f["topLayer"])
	vecs = vecs.transpose()
        #compute all pairwise similarities
        if sim == 'corr':
                m = 1-np.corrcoef(vecs)
                ALL.append([m[i,j] for i in range(m.shape[0]) for j in range(i+1,m.shape[0])])
        else:
                m = pdist(vecs, 'cosine')
                ALL.append(m)
        f.close()

	
if model == "ALL":
        m = np.corrcoef(ALL)
	print "Humans vs Words: ",m[1,0]
	print "Humans vs Pixels: ",m[1,2]
	print "Humans vs Sem: ",m[1,3]
	print "Humans vs Class: ",m[1,4]
	print "Humans vs SoftSem: ",m[1,5]
        print "Humans vs VGG: ",m[1,6]
	print 
	print "Words vs Sem: ",m[0,3]
	print "Pixels vs Sem: ",m[2,3]
	print 
	print "Words vs Class: ",m[0,4]
        print "Pixels vs Class: ",m[2,4]
	print
	print "Words vs VGG: ",m[0,6]
        print "Pixels vs VGG: ",m[2,6]

else:
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(m)

	# want a more natural, table-like display
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	plt.xlim(xmax=m.shape[0])
	plt.ylim(ymin=m.shape[1])

	plt.savefig(model+'.png')



