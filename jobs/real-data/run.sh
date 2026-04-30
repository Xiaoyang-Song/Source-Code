
# Reproducing results (seed 2024)
python run/test.py --dset="4B" --ckpt_name="6_h=6" --h=6 # don't know what this is

python run/test.py --dset="4B" --ckpt_name="1Dr-4D_1_h=6" --h=6 # Best

python run/test.py --dset="1Dr, 4D" --ckpt_name="4B_2_h=6" --h=6 # Best

# TWO CASES
python run/test.py --dset="4B" --ckpt_name="1Dr_h=6" --h=6 > log-2-4B.txt 
python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 > log-1-1Dr.txt

python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 > log-1Dr-8-0.1.txt 


python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.025 > checkpoint/realdata/log-1Dr-0.025.txt
python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.05 > checkpoint/realdata/log-1Dr-0.05.txt
python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.1 > checkpoint/realdata/log-1Dr-0.1.txt
python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.15 > checkpoint/realdata/log-1Dr-0.15.txt
python run/test.py --dset="1Dr" --ckpt_name="4B_h=6" --h=6 --alpha=0.2 > checkpoint/realdata/log-1Dr-0.2.txt