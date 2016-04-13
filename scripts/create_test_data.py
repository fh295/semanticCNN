# creates a small test set by subsampling from training data

from os import listdir, mkdir
from os.path import join, exists
from random import shuffle
import shutil

train_dir = '/local/filespace-2/fh295/DATA/train/'
val_dir = '/local/filespace-2/fh295/DATA/val/'
test_dir = '/local/filespace-2/fh295/DATA/test-small/'

TST_SIZE = 100

for cls in listdir(val_dir):
	cls_path = join(train_dir, cls)
	all_files = listdir(cls_path)
	shuffle(all_files)
	print(cls)
	
	new_dir = join(test_dir, cls)
	if not exists(new_dir):
        	mkdir(new_dir)
    
	
	for f in all_files[:TST_SIZE]:
		file_path = join(cls_path, f)
 		shutil.copy(file_path, new_dir)


