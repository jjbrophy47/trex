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
kernel_model_list=('klr' 'svm')
tree_kernel_list=('leaf_output' 'tree_output' 'leaf_path')

for rs in ${rs_list[@]}; do
    for kernel_model in ${kernel_model_list[@]}; do
        for tree_kernel in ${tree_kernel_list[@]}; do

            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=C_TX_$dataset \
                   --output=jobs/logs/cleaning/$dataset \
                   --error=jobs/errors/cleaning/$dataset \
                   jobs/cleaning/trex_runner.sh $dataset $tree_type $n_estimators \
                   $max_depth $check_pct $train_frac $kernel_model $tree_kernel $rs
        done
    done
done