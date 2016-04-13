import pdb
frequency = {}
with open('sorted.uk.word.unigrams','r') as f:
	for line in f.readlines():
		els = line.strip().split("\t")
		if int(els[0])<10:
			break
		frequency[els[1]] = els[0]
		

labels = {}
with open('labels.txt') as f:
	for line in f.readlines():
		els = line.strip().split("\t")
		synset = els[0]
		descr = els[1]
		candidates = []
		skip = False
		for el in descr.strip().split(','):
			el = el.strip().rstrip().lower()
			if not ' ' in el:
				if el in frequency:
					candidates.append((el,int(frequency[el])))
			else:
				last = el.strip().split(' ')[-1]
				if last in frequency:
					candidates.append((last, frequency[last]))
		if len(candidates)>0:
			s = sorted(candidates, key=lambda t: t[1], reverse=True)
			labels[synset] = (s[0][0], descr)
for s in labels:
	print s,"\t",labels[s][0],"\t",labels[s][1]
			
			
	
