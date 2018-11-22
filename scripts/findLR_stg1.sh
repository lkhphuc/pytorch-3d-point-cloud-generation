# python findLR_stg1.py --model NORMAL_STG1 --experiment sgd_plain \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --wd 0 \
# 	--startLR 1e-4 --endLR 2 --itersLR 100

# python findLR_stg1.py --model NORMAL_STG1 --experiment sgd_wd \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim sgd --wd 1e-4 \
# 	--startLR 1e-7 --endLR 10 --itersLR 100 \
# 	--gpu 0

# python findLR_stg1.py --model NORMAL_STG1 --experiment adam_plain \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --wd 0 \
# 	--startLR 1e-7 --endLR 10 --itersLR 100 \
# 	--gpu 0

# python findLR_stg1.py --model NORMAL_STG1 --experiment adam_wd \
# 	--chunkSize 100 --batchSize 100 \
# 	--optim adam --wd 1e-4 \
# 	--startLR 1e-7 --endLR 10 --itersLR 100 \
# 	--gpu 0

python findLR_stg1.py --model NORMAL_STG1 --experiment adam_trueWD \
	--chunkSize 100 --batchSize 100 \
	--optim adam --trueWD 1e-4 \
	--startLR 1e-7 --endLR 10 --itersLR 100 \
	--gpu 0
