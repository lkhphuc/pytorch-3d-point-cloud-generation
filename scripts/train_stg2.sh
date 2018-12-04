# python train_stg2.py --model ORIG_STG2 --experiment adam_trueWD \
# 	--loadPath ORIG_STG1_adam_trueWD \
# 	--chunkSize 100 --batchSize 32 \
# 	--endEpoch 500 --saveEpoch 20 \
# 	--optim adam --trueWD 1e-4 --lr 5e-3 \
# 	--gpu 0

# python train_stg2.py --model ORIG_STG2 --experiment sgd_trueWD \
	# --loadPath ORIG_STG1_sgd_trueWD \
	# --chunkSize 100 --batchSize 32 \
	# --endEpoch 500 --saveEpoch 20 \
	# --optim adam --trueWD 1e-4 --lr 1e-2 \
	# --gpu 0

# python train_stg2.py --model ORIG_STG2 --experiment adam_trueWD_cyclical \
# 	--loadPath ORIG_STG1_adam_trueWD \
# 	--chunkSize 100 --batchSize 32 \
# 	--endEpoch 500 --saveEpoch 20 \
# 	--optim adam --trueWD 1e-4 --lr 5e-4 \
# 	--lrSched cyclical --lrGamma 0.95 --lrMax 5e-1 --lrStep 55 \
# 	--gpu 0

# python train_stg2.py --model ORIG_STG2 --experiment adam_trueWD_restart1 \
# 	--loadPath ORIG_STG1_adam_trueWD \
# 	--chunkSize 32 --batchSize 32 \
# 	--optim adam --trueWD 1e-5 --lr 5e-3 \
# 	--lrSched restart --T_0 5 --T_mult 2 --lrBase 5e-2 \
# 	--gpu 0

# python train_stg2.py --model ORIG_STG2 --experiment sgd_trueWD_restart \
# 	--loadPath ORIG_STG1_sgd_trueWD_restart \
# 	--chunkSize 32 --batchSize 32 \
# 	--optim sgd --trueWD 1e-4 --lr 1e-1 \
# 	--lrSched restart --T_0 5 --T_mult 2 --lrBase 5e-3 \
# 	--gpu 1

# Continue training
python train_stg2.py --model ORIG_STG2 --experiment sgd_trueWD_restart_cont1 \
	--loadPath ORIG_STG1_sgd_trueWD_restart \
	--chunkSize 100 --batchSize 32 --saveEpoch 10 \
	--optim sgd --trueWD 1e-4 --lr 1e-1 \
	--lrSched restart --T_0 5 --T_mult 2 --lrBase 5e-3 \
	--gpu 1
