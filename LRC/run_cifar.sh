export FLAGS_fraction_of_gpu_memory_to_use=0.9
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

nohup env CUDA_VISIBLE_DEVICES=0 python -u train_mixup.py --batch_size=64 --auxiliary --mix_alpha=0.9 --model_id=0 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.01 --save_model_path=model_0 > lrc_model_0.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 python -u train_mixup.py --batch_size=64 --auxiliary --mix_alpha=0.6 --model_id=0 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.02 --save_model_path=model_1 > lrc_model_1.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 python -u train_mixup.py --batch_size=80 --auxiliary --mix_alpha=0.5 --model_id=1 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.015 --save_model_path=model_2 > lrc_model_2.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 python -u train_mixup.py --batch_size=80 --auxiliary --mix_alpha=0.6 --model_id=1 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.02 --save_model_path=model_3 > lrc_model_3.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=4 python -u train_mixup.py --batch_size=80 --auxiliary --mix_alpha=0.8 --model_id=1 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.03 --save_model_path=model_4 > lrc_model_4.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=5 python -u train_mixup.py --batch_size=64 --auxiliary --mix_alpha=0.5 --model_id=2 --cutout --lrc_loss_lambda=0.5 --weight_decay=0.0002 --learning_rate=0.015 --save_model_path=model_5 > lrc_model_5.log 2>&1 &

