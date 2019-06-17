# single GPU training
# python train.py -opt options/train/train_SRResNet.yml

# distributed training
# 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_EDVR_woTSA_M.yml --launcher pytorch