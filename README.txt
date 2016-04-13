get top1 predictions:

softsem
CUDA_VISIBLE_DEVICES=3 th eval.lua -modelPath imagenet/checkpoint/alexnet\,batchSize=32\,crit=class/MonApr1118\:43\:422016/model_29.t7 -crit softsem -data ../DATA/ > res.prediction.softsem

sem
CUDA_VISIBLE_DEVICES=3 th eval.lua -modelPath imagenet/checkpoint/MODELS_SEM/model_55.t7 -data ../DATA/ crit sem > res.prediction.sem


class
CUDA_VISIBLE_DEVICES=3 th eval.lua -modelPath imagenet/checkpoint/alexnet\,batchSize=32\,crit=class/MonApr1118\:43\:422016/model_50.t7 -data ../DATA/ crit class > res.prediction.class
