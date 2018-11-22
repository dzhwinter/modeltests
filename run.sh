export CUDA_VISIBLE_DEVICES=7
export FLAGS_nccl_dir=/usr/local/nccl/lib;
# export PYTHONPATH=/paddle/Paddle/bazel-pool/python/build/lib-python:$PYTHONPATH
export PYTHONPATH=/paddle/Paddle/bazel-memory/python/build/lib-python:$PYTHONPATH
	#--model=SE_ResNeXt152_32x4d \
rm out.log
GLOG_vmodule=build_strategy=3,analysis_var_pass=3,memory_reuse_pass=3,multi_devices_graph_print_pass=3 GLOG_logtostderr=1 FLAGS_fraction_of_gpu_memory_to_use=0.0 python train.py \
	--model=ResNet50 \
	--batch_size=32 \
	--use_gpu=True \
	--with_ir_mem_opt=False \
	--with_mem_opt=True 2>&1 | tee -a out.log
