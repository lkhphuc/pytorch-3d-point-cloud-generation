python train_stg1.py --model NORMAL_STG1 --experiment sgd_plain \
	--endEpoch 1000 \
	--chunkSize 100 --batchSize 100 \
	--optim sgd --lr 5e-2 \
	--gpu 1

# python train_stg1.py --model pcg_stg1  --experiment adam \
# 	--endEpoch 2000 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 \
# 	--gpu 1

# python train_stg1.py --model pcg_stg1 --experiment adam_exponential \
# 	--endEpoch 10000 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 --gpu 1

# python train_stg1.py --model pcg_stg1 --experiment adam_wd_expo \
# 	--endEpoch 2000 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 --wd 0.01 \
# 	--lrSched exponential --lrDecay 0.9 \
# 	--gpu 1

# python train_stg1.py --model pcg_stg1 --experiment adam_wd1e-2_expo \
# 	--endEpoch 2000 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 --wd 0.01 \
# 	--lrSched exponential --lrDecay 0.9 \
# 	--gpu 1

# python train_stg1.py --model pcg_stg1 --experiment adam_trueWD1e-2_expo \
# 	--endEpoch 500 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 --trueWD 1e-4\
# 	--lrSched exponential --lrDecay 0.90 \
# 	--gpu 1

# python train_stg1.py --model pcg_stg1 --experiment sgd \
# 	--endEpoch 1000 \
# 	--chunkSize 100 --batchSize 40 \
# 	--lr 1e-4 \
# 	--gpu 1
