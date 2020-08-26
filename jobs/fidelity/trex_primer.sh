dataset=$1
n_estimators=$2
max_depth=$3
mem=$4
time=$5
partition=$6

kernel_model_list=('klr' 'svm')
tree_kernel_list=('leaf_output' 'tree_output' 'leaf_path')

for kernel_model in ${kernel_model_list[@]}; do
    for tree_kernel in ${tree_kernel_list[@]}; do

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=F_TREX_$dataset \
               --output=jobs/logs/fidelity/$dataset \
               --error=jobs/errors/fidelity/$dataset \
               jobs/fidelity/trex_runner.sh $dataset $n_estimators \
               $max_depth $kernel_model $tree_kernel
