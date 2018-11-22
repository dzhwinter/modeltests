export CUDA_VISIBLE_DEVICES=7 export FLAGS_nccl_dir=/usr/local/nccl/lib;
# export PYTHONPATH=/paddle/Paddle/bazel-pool/python/build/lib-python:$PYTHONPATH
export PYTHONPATH=/paddle/Paddle/bazel-memory/python/build/lib-python:$PYTHONPATH
	#--model=SE_ResNeXt152_32x4d \
rm fluid.log
GLOG_vmodule=cfg_graph=3,memory_reuse_types=3,build_strategy=3,analysis_var_pass=3,memory_reuse_pass=3,multi_devices_graph_print_pass=3 GLOG_logtostderr=1 FLAGS_fraction_of_gpu_memory_to_use=0.0 python train.py \
	--model=ResNet50 \
	--batch_size=32 \
	--use_gpu=True \
	--with_ir_mem_opt=True \
	--with_mem_opt=False 2>&1 | tee -a fluid.log
