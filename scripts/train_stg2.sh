python train_stg2.py --model pcg_stg2  --experiment adam \
	--loadPath pcg_stg1 --load 0 \
	--endEpoch 500 \
	--chunkSize 30 --batchSize 30 \
	--lr 1e-4 \
	--gpu 0

python train_stg2.py --model pcg_stg2 --experiment adam_exponential \
	--loadPath pcg_stg1 --load 0 \
	--endEpoch 500 \
	--chunkSize 30 --batchSize 30 \
	--lr 1e-4 \
	--gpu 0

python train_stg2.py --model pcg_stg2 --experiment adam_wd_expo \
	--loadPath pcg_stg1 --load 0 \
	--endEpoch 500 \
	--chunkSize 30 --batchSize 30 \
	--lr 1e-4 --wd 0.01 \
	--lrSched exponential --lrDecay 0.9 \
	--gpu 0

python train_stg2.py --model pcg_stg2 --experiment adam_wd1e-2_expo \
	--loadPath pcg_stg1 --load 0 \
	--endEpoch 500 \
	--chunkSize 100 --batchSize 40 \
	--lr 1e-4 --wd 0.01 \
	--lrSched exponential --lrDecay 0.9 \
	--gpu 0
