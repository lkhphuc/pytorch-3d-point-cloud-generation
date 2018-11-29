# python train_stg1.py --model ORIG_STG1 --experiment adam_trueWD \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --trueWD 1e-4 --lr 5e-3 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment sgd_trueWD \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --trueWD 1e-4 --lr 5e-2 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment adam_wd \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --wd 1e-4 --lr 5e-3 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment sgd_wd \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --wd 1e-4 --lr 5e-2 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment adam_plain \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --lr 5e-3 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment sgd_plain \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd  --lr 1e-1 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment adam_annealing \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --trueWD 1e-4 --lr 5e-2 \
# 	--lrSched annealing --lrGamma 0.95 \
# 	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment sgd_trueWD_momentum_restart \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --trueWD 1e-4 --momentum 0.9 --lr 1e-1 \
# 	--lrSched restart --T_0 5 --T_mult 2 --lrBase 1e-3 \
# 	--gpu 1

python train_stg1.py --model ORIG_STG1 --experiment adam_trueWD_restart \
	--endEpoch 1000 \
	--chunkSize 100 --batchSize 100 \
	--optim adam --trueWD 1e-4 --lr 1e-2 \
	--lrSched restart --T_0 5 --T_mult 2 --lrBase 1e-4 \
	--gpu 1

# python train_stg1.py --model ORIG_STG1 --experiment sgd_trueWD_restart \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --trueWD 1e-4 --lr 1e-1 \
# 	--lrSched restart --T_0 10 --T_mult 2 --lrBase 5e-5 \
# 	--gpu 1

