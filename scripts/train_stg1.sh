# python train_stg1.py --model NORMAL_STG1 --experiment sgd_plain \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --lr 5e-2 \
# 	--gpu 1

# python train_stg1.py --model NORMAL_STG1 --experiment sgd_wd \
# 	--endEpoch 1000 \
# 	--chunkSize 300 --batchSize 300 \
# 	--optim sgd --wd 1e-4 --lr 5e-2 \
# 	--gpu 0

# python train_stg1.py --model NORMAL_STG1 --experiment adam_plain \
# 	--endEpoch 1000 \
# 	--chunkSize 300 --batchSize 300 \
# 	--optim adam --wd 0 --lr 5e-3 \
# 	--gpu 1

# python train_stg1.py --model NORMAL_STG1 --experiment adam_wd \
# 	--endEpoch 1000 \
# 	--chunkSize 300 --batchSize 300 \
# 	--optim adam --wd 1e-4 --lr 5e-3 \
# 	--gpu 0

# python train_stg1.py --model NORMAL_STG1 --experiment adam_trueWD \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --trueWD 1e-4 --lr 5e-2 \
# 	--gpu 0

# python train_stg1.py --model ORIG_STG1 --experiment adam_trueWD \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --trueWD 1e-4 --lr 1e-3 \
# 	--gpu 1

python train_stg1.py --model ORIG_STG1 --experiment sgd_trueWD \
	--endEpoch 1000 \
	--chunkSize 100 --batchSize 100 \
	--optim sgd --trueWD 1e-4 --lr 5e-3 \
	--gpu 1
