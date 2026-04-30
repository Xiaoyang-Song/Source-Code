#!/bin/bash
# Sensitivity analysis for window_size and h

for sigma in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5
do
    echo "Running sigma=$sigma"
    # Generate data
    python jobs/simulation/gen_data.py \
    --data_dir="simulation-circle-crack" \
    --data_dir_2="simulation-circle-crack" \
    --n=10000 \
    --sp_type="2fc" \
    --sigma=$sigma
    
    for h in 2 4 6 8 10
    do
        # Run CLS training
        python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="CLS" \
        --h=$h \
        --n_abn=250
        
        # Run AE training
        python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="AE" \
        --h=$h
        
        # Run SIMCLR training
        python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="SIMCLR" \
        --h=$h
        
        # Run SIMCLR training
        python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="SIMCLR-CLIP" \
        --h=$h
        
        for window_size in 15 20 25 30 35 40
        do
            echo "Running window_size=$window_size, h=$h, sigma=$sigma"
            
            save_dir="checkpoint/simulation-circle-crack/sensitivity_analysis/${h}-${window_size}"
            mkdir -p $save_dir
            
            # Run DCCA training (window size matters)
            python jobs/simulation/automate.py \
            --data_dir='simulation-circle-crack' \
            --ckpt_dir='simulation-circle-crack/SA_models' \
            --flag="DCCA" \
            --h=$h \
            --window_size=$window_size \
            > ${save_dir}/dcca-training-sigma${sigma}.log
            
            # Run baseline test
            python jobs/simulation/test_baseline.py \
            --data_dir='simulation-circle-crack' \
            --ckpt_dir='simulation-circle-crack/SA_models' \
            --h=$h \
            --window_size=$window_size \
            --n_abn=250 \
            > ${save_dir}/results-sigma${sigma}.log
            
            echo "Done window_size=$window_size, h=$h, sigma=$sigma"
        done
        echo "Finished h=$h, sigma=$sigma"
    done
    echo "Finished sigma=$sigma"
done
