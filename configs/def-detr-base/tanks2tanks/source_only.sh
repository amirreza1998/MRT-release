N_GPUS=1
BATCH_SIZE=1
# DATA_ROOT="D:/kazemloo/1-domain_adaptation/docker_code/app2/data"
DATA_ROOT="/app/app2/data"
OUTPUT_DIR=/app/app2/data/outputs/def-detr-base/tanks2tanks/source_only

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun \
# --rdzv_endpoint localhost:26507 \
# --nproc_per_node=${N_GPUS} \
CUDA_VISIBLE_DEVICES=0 python main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset tanks \
--target_dataset tanks \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 50 \
--epoch_lr_drop 60 \
--mode single_domain \
--output_dir ${OUTPUT_DIR}

