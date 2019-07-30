export FLAGS_fraction_of_gpu_memory_to_use=0.6
nohup env CUDA_VISIBLE_DEVICES=0 python -u test_mixup.py --batch_size=64 --auxiliary --model_id=0 --pretrained_model=model_0/final/ --dump_path=paddle_predict/prob_test_0.pkl > lrc_test_0.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 python -u test_mixup.py --batch_size=64 --auxiliary --model_id=0 --pretrained_model=model_1/final/ --dump_path=paddle_predict/prob_test_1.pkl > lrc_test_1.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 python -u test_mixup.py --batch_size=80 --auxiliary --model_id=1 --pretrained_model=model_2/final/ --dump_path=paddle_predict/prob_test_2.pkl > lrc_test_2.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 python -u test_mixup.py --batch_size=80 --auxiliary --model_id=1 --pretrained_model=model_3/final/ --dump_path=paddle_predict/prob_test_3.pkl > lrc_test_3.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=4 python -u test_mixup.py --batch_size=80 --auxiliary --model_id=1 --pretrained_model=model_4/final/ --dump_path=paddle_predict/prob_test_4.pkl > lrc_test_4.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=5 python -u test_mixup.py --batch_size=64 --auxiliary --model_id=2 --pretrained_model=model_5/final/ --dump_path=paddle_predict/prob_test_5.pkl > lrc_test_5.log 2>&1 &

