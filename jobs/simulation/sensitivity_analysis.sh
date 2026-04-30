for window_size in 20 25 30 35
do
  for h in 4 6 8 10
  do
    echo "Running window_size=$window_size, h=$h"

    save_dir="checkpoint/simulation-circle-crack/sensitivity_analysis/${h}-${window_size}"
    mkdir -p $save_dir

    for sigma in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
    do
      echo "Running sigma=$sigma"

      # Generate data
      python jobs/simulation/gen_data.py \
        --data_dir="simulation-circle-crack" \
        --data_dir_2="simulation-circle-crack" \
        --n=10000 \
        --sp_type="2fc" \
        --sigma=$sigma

      # Run DCCA training
      python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="DCCA" \
        --h=$h \
        --window_size=$window_size \
        > ${save_dir}/dcca-training-sigma${sigma}.log

      # Run CLS training
      python jobs/simulation/automate.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --flag="CLS" \
        --h=$h \
        --window_size=$window_size \
        --n_abn=250

      # Run baseline test
      python jobs/simulation/test_baseline.py \
        --data_dir='simulation-circle-crack' \
        --ckpt_dir='simulation-circle-crack/SA_models' \
        --h=$h \
        --window_size=$window_size \
        --n_abn=250 \
        > ${save_dir}/results-sigma${sigma}.log

      echo "Done sigma=$sigma"
    done

    echo "Finished window_size=$window_size, h=$h"
  done
done