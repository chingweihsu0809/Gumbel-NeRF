#!/bin/bash

#$ -l f_node=16
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
        --master_port=12345 \
        --nproc_per_node=4 \
        --nnodes=$nnodes \
        --node_rank=$node_rank \
        --master_addr=$master_addr \
        -m src.test_time_opt --exp_name ./exp/codedswitchnerf/tto \
                             --ckpt_path ./exp/codedswitchnerf/train_fine/0/models/45000.pt \
                             --dataset_path ./data/srn_cars \
                             --task "tto" \
                             --img_index=64 \
                             --use_moe \
                             --switch_config src/models/switch_model/config/srn_cars.yaml \
                             --moe_expert_type=expertmlp \
                             --moe_train_batch \
                             --moe_test_batch \
                             --moe_capacity_factor=1.0 \
                             --moe_l_aux_wt=0.0005 \
                             --use_moe_external_gate \
                             --use_sigma_noise \
                             --use_gate_input_norm  \
                             --batch_prioritized_routing \
                             --sigma_noise_std=1.0 \
                             --image_pixel_batch_size=4096 \
                             --model_chunk_size=122880 \
                             --lr=2e-2 \
                             --appearance_dim=0 \
                             --latent_dim=256 \
                             --train_iterations=201 \
                             --white_bkgd \
                             --item_files_postfix "_multi_nocrop" \
                             --no_load_latent \
                             --latent_init="zero" \
                             --latent_src="cars_test" \
                             --find_unused_parameters
                             
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
