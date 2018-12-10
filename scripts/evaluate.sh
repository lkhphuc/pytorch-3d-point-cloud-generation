# python evaluate.py --model ORIG_STG2 --experiment adam_trueWD_cyclical \
# 	--loadPath ORIG_STG2_adam_trueWD_cyclical \
# 	--chunkSize 32 --batchSize 32 \
# 	--gpu 0

python evaluate.py --model ORIG_STG2 --experiment sgd_trueWD_restart_cont1 \
	--loadPath ORIG_STG2_sgd_trueWD_restart_cont1 \
	--chunkSize 32 --batchSize 32 \
	--gpu 0
