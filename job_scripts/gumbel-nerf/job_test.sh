#!/bin/bash

#$ -l f_node=32
#$ -l h_rt=8:00:00
#$ -j y
#$ -cwd

NNODES=`wc -l < $PE_HOSTFILE`
MASTER_ADDR=`head -n 1 $PE_HOSTFILE | cut -d " " -f 1`

echo "NNODE:$NNODES"
echo "MASTER_ADDR:$MASTER_ADDR"

remote_commands() {
    local nnodes=$1
    local node_rank=$2
    local master_addr=$3

    export NCCL_DEBUG=INFO
    export NCCL_BUFFSIZE=1048576

    module load cuda/11.6.1 python/3.10.2 nccl/2.17.1 gcc/11.2.0
    module list
    
    export WORKSPACE="/path/to/workspace"
    cd $WORKSPACE/Gumbel-NeRF
    source "$WORKSPACE/python-venv/switch-nerf_py310/bin/activate"

    echo "Executing task on $(hostname)"
    echo "nnodes: $nnodes"
    echo "node_rank: $node_rank"
    echo "master_addr: $master_addr"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
        --use_env \
        --master_port=12346 \
        --nproc_per_node=4 \
        --nnodes=$nnodes \
        --node_rank=$node_rank \
        --master_addr=$master_addr \
        -m src.eval_image --exp_name ./exp/gumbelnerf/test \
                          --ckpt_path /gs/hs0/tga-RLA/21M38136/Switch-NeRF_srn/exp/gumbelnerf_srn_multi_0912_CodeNeRFstyleSig_W0.5/uni_lr13e-4_Tmax20/tto_fine/Tmax2_eta_max2min0.5/bs4096_lr2e-2/0/models/200.pt \
                          --dataset_path data/srn_cars \
                          --task "test" \
                          --exclude_img_index=64 \
                          --use_gumbel \
                          --gumbel_config src/models/gumbel_model/configs/multi_unihead_cn_W0.5.json \
                          --model_chunk_size=131072 \
                          --appearance_dim=0 \
                          --latent_dim=256 \
                          --eta_min=0.5 \
                          --white_bkgd \
                          --item_files_postfix "_multi_nocrop" \
                          --latent_src="cars_test" 

}

task() {
    local node_rank=$1
    local server=$(awk "NR==$node_rank{print \$1}" "$PE_HOSTFILE")
    
    ssh "user@$server" "$(declare -f remote_commands); remote_commands $NNODES $((node_rank - 1)) $MASTER_ADDR"

}

cat $PE_HOSTFILE

for ((node_rank=1; node_rank<=NNODES; node_rank++))
do
    task "$node_rank" &
done

wait 
exit
