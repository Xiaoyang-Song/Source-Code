
# Reproducing results (seed 2024)

# Image AE pretraining
# python run/img_ae.py --dset=4B --lr=0.001 --lr_step=2000 --n_epochs=500 --bsz_tri=256 --bsz_val=256 --h=6 --tag=1-h=6

# DCCA training
# python run/main_dcca.py --dset=4B --lr=0.00005 --n_epochs=500 --bsz_tri=8 --bsz_val=8 --h=6 --normalize --n_log=1 --tag=h=6 --ct_ckpt_name=4B_1-h=6

# Online monitoring evaluation
# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 > log-real.txt


# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.025 > checkpoint/realdata/log-1Dr-0.025.txt
# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.05 > checkpoint/realdata/log-1Dr-0.05.txt
# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.1 > checkpoint/realdata/log-1Dr-0.1.txt
# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.15 > checkpoint/realdata/log-1Dr-0.15.txt
# python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.2 > checkpoint/realdata/log-1Dr-0.2.txt