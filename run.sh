CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/liming/chenhan/project/pipe/result/DL-based-findcenter/simple-classification-frame/main.py \
    --batch_size 16 \
    --epochs 100 \
    --milestones 40 70 \
    --logs_dir ../824_try_1