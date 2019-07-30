export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=1.
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train_imagenet.py --batch_size=64 > imagenet.log 2>&1 &

