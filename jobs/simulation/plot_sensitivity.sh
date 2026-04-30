# Fixed p
python jobs/simulation/plot_sensitivity_new.py \
    --n-values 25 \
    --plot-combinations \
    --combo-method DCCA \
    --fixed-p 6
# Fixed n
python jobs/simulation/plot_sensitivity_new.py \
  --plot-combinations \
  --combo-method DCCA \
  --fixed-n 35 \
  --p-values 2,4,6,8,10