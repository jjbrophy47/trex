dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
check_pct=$5
train_frac=$6
mem=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=C_P_$dataset \
           --output=jobs/logs/cleaning/$dataset \
           --error=jobs/errors/cleaning/$dataset \
           jobs/cleaning/proto_runner.sh $dataset $tree_type $n_estimators \
           $max_depth $check_pct $train_frac $rs
done