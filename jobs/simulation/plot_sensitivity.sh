python jobs/simulation/plot_sensitivity_new.py --p-values 2,4,6,8,10 --n-values 15,20,25,30,35 --plot-best-baseline

python jobs/simulation/plot_sensitivity_new.py --p-values 4,6,8 --n-values 20,25,30,35 --plot-best-baseline


# Fixed p
python jobs/simulation/plot_sensitivity_new.py \
    --n-values 15,20,25,30,35 \
    --plot-combinations \
    --combo-method DCCA \
    --fixed-p 6
# Fixed n
python jobs/simulation/plot_sensitivity_new.py \
  --plot-combinations \
  --combo-method DCCA \
  --fixed-n 35 \
  --p-values 2,4,6,8,10