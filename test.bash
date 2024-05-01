# configfile="dominik_eval_lidar_occ_fisheye"
# checkpoint="/storage/user/muhled/outputs/mvbts/kitti-360/simple_multi_view_head_backend-None-1_20231030-163439/training_checkpoint_200000.pt"
configfile="dominik_eval_lidar_occ_base"
checkpoint="/storage/user/muhled/outputs/mvbts/kitti-360/simple_multi_view_head_w_attn_backend-None-1_20231101-094904/training_checkpoint_200000.pt"
new_architecture="true"
# no spaces are important
# encoder_ids="[0]" # mono 
# encoder_ids="[0,1]" # temporal 
# encoder_ids="[0,2]" # stereo 
encoder_ids="[0,1,2,3]" # stereo-temporal 
# encoder_ids="[0,1,2,3,4,5,6,7]" # full 

python eval.py -cn $configfile ++model_conf.encoder_ids=$encoder_ids ++checkpoint=$checkpoint ++new_model_architecture=$new_architecture