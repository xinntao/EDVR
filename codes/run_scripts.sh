# single GPU training
# python train.py -opt options/train/train_SRResNet.yml

# distributed training
# 4 GPUs
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/train_EDVR_woTSA_M.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/train_EDVR_M.yml --launcher pytorch

# train with Youku dataset 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 train.py -opt options/train/train_EDVR_woTSA_Youku.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4322 train.py -opt options/train/train_EDVR_Youku.yml --launcher pytorch

# train VSREDUN with Youku dataset 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 train.py -opt options/train/train_VSREDUN_woTSA_Youku.yml --launcher pytorch