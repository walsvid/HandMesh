phase='train'
exp_name='cmr_sg_train'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_sg'
python main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx 0 \