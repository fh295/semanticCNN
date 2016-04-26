import cPickle
import pdb

with open('mapping.txt') as inp:
    D = inp.read().splitlines()
    D_split = [x.split('\t') for x in D]
    words = [x[1].strip() for x in D_split]

with open('/home/fh295/Documents/Deep_learning_Bengio/arctic/defgen/D_cbow_pdw_8B.pkl') as inp:
    D = cPickle.load(inp)


W = {}
for w in words:
    try:
        W[w] = D[w]
    except:
        print 'we do not have this word: %s' % (w)
        pass

with open('semantic_vector_dict.pkl','w') as out:
    cPickle.dump(W,out)

