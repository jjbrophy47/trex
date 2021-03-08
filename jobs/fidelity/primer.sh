dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
mem=$6
time=$7
partition=$8

tree_kernel='leaf_output'
metric='mse'
surrogate_list=('klr' 'svm' 'knn')
rs_list=(1 2 3 4 5)

for surrogate in ${surrogate_list[@]}; do
    for rs in ${rs_list[@]}; do

        # if [ $tree_kernel = 'feature_path' ] || [ $tree_kernel = 'feature_output' ]; then
        #     preprocessing='standard'
        # else
        #     preprocessing=$3
        # fi

        job_name="F_${dataset}_${model}_${surrogate}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/fidelity/$job_name \
               --error=jobs/errors/fidelity/$job_name \
               jobs/fidelity/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $surrogate $tree_kernel $metric $rs
    done
done
