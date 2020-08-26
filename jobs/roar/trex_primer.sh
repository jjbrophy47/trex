dataset=$1
n_estimators=$2
max_depth=$3
mem=$4
time=$5
partition=$6

rs_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
kernel_model_list=('klr')
tree_kernel_list=('leaf_output' 'tree_output' 'leaf_path')

for rs in ${rs_list[@]}; do
    for kernel_model in ${kernel_model_list[@]}; do
        for tree_kernel in ${tree_kernel_list[@]}; do

            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=R_TREX_$dataset \
                   --output=jobs/logs/roar/$dataset \
                   --error=jobs/errors/roar/$dataset \
                   jobs/roar/trex_runner.sh $dataset $n_estimators \
                   $max_depth $kernel_model $tree_kernel $rs
