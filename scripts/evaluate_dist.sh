# python evaluate_dist.py --model ORIG_STG2 --experiment adam_trueWD_cyclical \
# 	--loadPath ORIG_STG2_adam_trueWD_cyclical \
# 	--chunkSize 32 --batchSize 32 

python evaluate_dist.py --model ORIG_STG2 --experiment sgd_trueWD_restart_cont1 \
	--loadPath ORIG_STG2_sgd_trueWD_restart_cont1 \
	--chunkSize 32 --batchSize 32 \
	--gpu 1

# python evaluate_dist.py --model ORIG_STG2 --experiment orig_tf \
# 	--loadPath "~/3D-point-cloud-generation/results_0/orig-ft_it100000" \
# 	--chunkSize 32 --batchSize 32 \
# 	--gpu 1
