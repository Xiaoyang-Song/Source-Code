



python run/main_dl_baseline.py --dset=4B --lr=0.001 --flag="AE" --n_epochs=100 --bsz_tri=8 --bsz_val=8 --h=6 --normalize --n_log=1 --tag=h=6

python run/main_dl_baseline.py --dset=4B --lr=0.001 --flag="SIMCLR" --n_epochs=100 --bsz_tri=8 --bsz_val=8 --h=6 --normalize --n_log=1 --tag=h=6

python run/main_dl_baseline.py --dset=4B --lr=0.001 --flag="SIMCLR-CLIP" --n_epochs=100 --bsz_tri=8 --bsz_val=8 --h=6 --normalize --n_log=1 --tag=h=6