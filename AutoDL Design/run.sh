export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
CUDA_VISIBLE_DEVICES=1 python -u main.py > a.log 2>&1
