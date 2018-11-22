export CUDA_VISIBLE_DEVICES=7
export FLAGS_nccl_dir=/usr/local/nccl/lib;
# export PYTHONPATH=/paddle/Paddle/bazel-pool/python/build/lib-python:$PYTHONPATH
export PYTHONPATH=/paddle/Paddle/bazel-memory/python/build/lib-python:$PYTHONPATH
	#--model=SE_ResNeXt152_32x4d \
GLOG_vmodule=build_strategy=3,analysis_var_pass=3,multi_devices_graph_print_pass=3 GLOG_logtostderr=1 python casecade_net.py \
	     --with_ir_mem_opt=False \
       2>&1 | tee -a casecade_net.log
