# Test circle-crack with noises

for sigma in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5
do
    echo "Running with sigma=$sigma"
    
    # Generate data with the current sigma
    python jobs/simulation/gen_data.py \
    --data_dir="simulation-circle-crack" \
    --data_dir_2="simulation-circle-crack" \
    --n=10000 \
    --sp_type="2fc" \
    --sigma=$sigma
    
    # Run automate.py
    python jobs/simulation/automate.py \
    --data_dir='simulation-circle-crack' \
    --flag="DCCA" \
    --h=6 \
    > checkpoint/simulation-circle-crack/dcca-training-sigma${sigma}.log
    
    # Run CLS training
    python jobs/simulation/automate.py --data_dir='simulation-circle-crack' \
    --flag="CLS" --h=6 --n_abn=250
    
    # Run test_baseline.py
    python jobs/simulation/test_baseline.py \
    --data_dir='simulation-circle-crack' \
    --h=6 \
    --n_abn=250 \
    > checkpoint/simulation-circle-crack/results-sigma${sigma}.log
    
    echo "Done with sigma=$sigma"
done

