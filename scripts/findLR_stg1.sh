python findLR_stg1.py --model ORIG_STG1 --experiment adam_trueWD \
	--chunkSize 100 --batchSize 100 \
	--optim adam --trueWD 1e-4 \
	--startLR 1e-5 --endLR 1 --itersLR 25 \
	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment sgd_trueWD \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --trueWD 1e-4 \
# 	--startLR 1e-7 --endLR 10 --itersLR 50 \
# 	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment adam_wd \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --wd 1e-4 \
# 	--startLR 1e-7 --endLR 10 --itersLR 50 \
# 	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment sgd_wd \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --wd 1e-4 \
# 	--startLR 1e-7 --endLR 10 --itersLR 50 \
# 	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment adam_plain \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam \
# 	--startLR 1e-7 --endLR 10 --itersLR 50 \
# 	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment sgd_plain \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd \
# 	--startLR 1e-7 --endLR 10 --itersLR 50 \
# 	--gpu 0

# python findLR_stg1.py --model ORIG_STG1 --experiment sgd_trueWD_momentum \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --momentum 0.9 \
# 	--startLR 1e-5 --endLR 1 --itersLR 20 \
# 	--gpu 0
