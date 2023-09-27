import configargparse
import yaml


def get_opts_base():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
        
    ## dataset
    parser.add_argument('--chunk_paths', type=str, nargs='+', default=None,
                        help="""scratch directory to write shuffled batches to when training using the filesystem dataset. 
                                Should be set to a non-existent path when first created, and can then be reused by subsequent training runs once all chunks are written""")
    parser.add_argument('--num_chunks', type=int, default=200,
                        help='number of shuffled chunk files to write to disk. Each chunk should be small enough to fit into CPU memory')
    parser.add_argument('--generate_chunk', default=False, action='store_true', help='only generate chunks')
    parser.add_argument("--shuffle_chunk", action='store_true', default=False, help='random shuffle the order of chunks before loading')
    parser.add_argument("--data_loader_num_workers", type=int, default=1,help='num_workers arg in data loader')
    parser.add_argument('--disk_flush_size', type=int, default=10000000)
    parser.add_argument('--task', type=str, choices=['train', 'test', 'tto', 'test_on_train', 'fastrun'], default="train")
    parser.add_argument('--no_load_latent', dest='load_latent', default=True, action='store_false')
    parser.add_argument('--item_files_postfix', type=str, default="_multi")
    parser.add_argument('--no_crop_img', dest='crop_img', default=True, action='store_false')
    parser.add_argument('--img_index', type=int, nargs='+', default=[])    ## sideview: 15, 54, 93, 132; topview: 250; refview in codenerf: 64; [64, 104]
    parser.add_argument('--exclude_img_index', type=int, nargs='+', default=[])    
    parser.add_argument('--save_img_index', type=int, nargs='+', default=list(range(0,251,1)))
    parser.add_argument('--latent_dim', type=int, default=0)
    parser.add_argument('--latent_init', default="random", choices=['random', 'zero', 'train_avg'], type=str)    
    parser.add_argument('--latent_src', default="cars_train", type=str)   
    

    ## ckpt related
    parser.add_argument('--ckpt_path', type=str, default=None, help='path towards serialized model checkpoint')
    parser.add_argument('--find_unused_parameters', default=False, action='store_true', help='whether using moe nerf')
    parser.add_argument('--no_resume_ckpt_state', dest='resume_ckpt_state', default=True, action='store_false')


    ## memory related
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--image_pixel_batch_size', type=int, default=64 * 1024,
                        help='number of pixels to evaluate per split when rendering validation images')
    parser.add_argument('--model_chunk_size', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument("--compute_memory", action='store_true', default=False, 
                        help='log the max memory in each step')


    ##global
    parser.add_argument('--no_amp', dest='amp', default=True, action='store_false')
    parser.add_argument('--detect_anomalies', default=False, action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument("--amp_use_bfloat16", action='store_true', default=False, help='use bfloat16 in amp of pytorch to see if still nan')
    parser.add_argument("--no_optimizer_schedulers", action='store_true', default=False, help='diable learning scheduler')
    parser.add_argument("--disable_check_finite", action='store_true', default=False, help='disable check_finite after forward for efficiency and stable training')

    
    ## train related
    parser.add_argument('--train_iterations', type=int, default=500000, help='training iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--train_every', type=int, default=1, help='if set to larger than 1, subsamples each n training images')
    parser.add_argument('--val_interval', type=int, default=500001, help='validation interval')
    parser.add_argument('--ckpt_interval', type=int, default=100, help='checkpoint interval')
    parser.add_argument("--i_print",   type=int, default=1000, help='frequency of console printout and metric logging')


    ## rays/rendering related
    parser.add_argument('--near', type=float, default=0.8, help='ray near bounds')
    parser.add_argument('--far', type=float, default=1.8, help='ray far bounds.')
    parser.add_argument('--coarse_samples', type=int, default=96, help='number of coarse samples')
    parser.add_argument('--fine_samples', type=int, default=0, help='number of additional fine samples')
    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument("--use_sigma_noise", action='store_true', default=False, help='use noise for sigma')
    parser.add_argument("--sigma_noise_std", type=float, default=1.0, help='std of noise for sigma')
    parser.add_argument('--no_center_pixels', dest='center_pixels', default=True, action='store_false',
                        help='do not shift pixels by +0.5 when computing ray directions')
    parser.add_argument('--no_shifted_softplus', dest='shifted_softplus', default=True, action='store_false',
                        help='use ReLU instead of shifted softplus activation')
    parser.add_argument("--white_bkgd", action='store_true', default=False, help='set to render synthetic data on a white bkgd')
    
    
    ## Model common
    parser.add_argument("--expert_num", type=int, default=4, help='number of expert')
    parser.add_argument('--appearance_dim', type=int, default=48, help='dimension of appearance embedding vector (set to 0 to disable)')
    parser.add_argument('--pos_xyz_dim', type=int, default=12, help='frequency encoding dimension applied to xyz position')
    parser.add_argument('--pos_dir_dim', type=int, default=4, help='frequency encoding dimension applied to view direction (set to 0 to disable)')
    
    
    ## For Gumbel-NeRF
    parser.add_argument('--use_gumbel', default=False, action='store_true')
    parser.add_argument('--gumbel_config', default="", type=str)
    parser.add_argument("--eta_max", type=float, default=10)
    parser.add_argument("--eta_min", type=float, default=0.5)
    parser.add_argument("--T_max", type=float, default=100000)
    parser.add_argument("--T_init", type=float, default=None)
    parser.add_argument("--latent_lr_gain", type=float, default=1.)
    parser.add_argument("--reg_wt", type=float, default=0, help='reg loss')   
    
    
    ## For Coded Switch-NeRF
    parser.add_argument('--switch_config', is_config_file=True)
    parser.add_argument('--switch_model', type=yaml.safe_load)   
    parser.add_argument('--use_moe', default=False, action='store_true',
                        help='whether using switch nerf')
    parser.add_argument("--moe_l_aux_wt", type=float, default=1e-2, 
                        help='l_aux_wt of tutel moe')    
    parser.add_argument("--moe_capacity_factor", type=float, default=1.25, 
                        help='capacity_factor of tutel moe')    
    parser.add_argument("--moe_use_residual", default=False, action='store_true', 
                        help='use residual moe')
    parser.add_argument("--moe_expert_type", type=str, default='expertmlp', 
                        help='expert type of the moe')
    parser.add_argument("--moe_train_batch", default=False, action='store_true', 
                        help='use batched moe for training')    
    parser.add_argument("--moe_test_batch", default=False, action='store_true', 
                        help='use batched moe for testing')
    parser.add_argument("--expertmlp2seqexperts", action='store_true', default=False, 
                        help='convert state_dict of expertmlp to state_dict of seqexperts')
    parser.add_argument('--no_expert_parallel', default=True, action='store_true',
                        help='do not use expert parallel in moe')    
    parser.add_argument('--no_use_balance_loss', dest='use_balance_loss', default=True, action='store_false', 
                        help='not use load balance loss in moe')
    parser.add_argument("--batch_prioritized_routing", action='store_true', default=False, 
                        help='use batch_prioritized_routing in moe gate, extract_critical_nobatch currently not support this')
    parser.add_argument('--affine_appearance', default=False, action='store_true',
                        help='set to true to use affine transformation for appearance instead of latent embedding')
    parser.add_argument("--use_moe_external_gate", action='store_true', default=False, 
                        help='''use a small network as a gate in MoE layers.''')
    parser.add_argument("--use_gate_input_norm", action='store_true', default=False, 
                        help='use norm layer for gate input, support layernorm and batchnorm')
    parser.add_argument("--gate_noise", type=float, default=-1.0,
                        help='std of gate noise when use load_importance_los')
    parser.add_argument("--use_load_importance_loss", action='store_true', default=False, 
                        help='use load_importance_loss, gate_noise should above zero')
    parser.add_argument("--compute_balance_loss", action='store_true', default=False, 
                        help='compute_balance_loss when use load_importance_loss, for comparison')
    parser.add_argument("--dispatcher_no_score", action='store_true', default=False, 
                        help='do not multiply socre on moe output, only use for testing')
    parser.add_argument("--dispatcher_no_postscore", action='store_true', default=False, 
                        help='multiply gate score before feeded into moe')


    # returns
    parser.add_argument("--moe_return_gates", default=False, action='store_true',
                        help='return gates index for each point')
    parser.add_argument("--moe_return_gate_logits", default=False, action='store_true',
                        help='return gate logits after wg and before softmax')
    parser.add_argument("--return_pts", action='store_true', default=False,
                        help='return the sample points out of render function')
    parser.add_argument("--return_pts_rgb", action='store_true', default=False,
                        help='return the color of sample points out of render function')
    parser.add_argument("--return_pts_alpha", action='store_true', default=False,
                        help='return the alpha of sample points out of render function')
    parser.add_argument('--render_test_points_typ', type=str, nargs='+', default=["coarse"], 
                        help='point cloud from the coarse sample or fine sample or both, currently only support coarse')
    parser.add_argument("--render_test_points_sample_skip", type=int, default=1, 
                        help='skip number for point samples of each pixel to reduce the point cloud size')
    parser.add_argument("--render_test_points_image_num", type=int, default=1, 
                        help='image number for color point clouds indicating expert ids')
    parser.add_argument("--return_pts_class_seg", default=False, action='store_true',
                        help='return the colored segmentation of each centroid')
    parser.add_argument("--return_sigma", default=False, action='store_true',
                        help='return sigma in rendering results')
    parser.add_argument("--return_alpha", default=False, action='store_true',
                        help='return alpha in rendering results')
         
    
    return parser
