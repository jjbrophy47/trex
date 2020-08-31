dataset=$1
n_estimators=$2
max_depth=$3
check_pct=$4
train_frac=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=C_MD_$dataset \
           --output=jobs/logs/cleaning/$dataset \
           --error=jobs/errors/cleaning/$dataset \
           jobs/cleaning/mmd_runner.sh $dataset $n_estimators \
           $max_depth $check_pct $train_frac $rs
done