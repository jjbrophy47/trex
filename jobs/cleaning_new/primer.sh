dataset=$1
model=$2
n_estimators=$3
max_depth=$4
mem=$5
time=$6
partition=$7

method_list=(
             'klr_og_tree_output_alpha'
             'klr_og_tree_output_sim'
             'klr_og_leaf_path_alpha'
             'klr_og_leaf_path_sim'
             'klr_og_weighted_leaf_path_alpha'
             'klr_og_weighted_leaf_path_sim'
             'klr_og_weighted_feature_path_alpha'
             'klr_og_weighted_feature_path_sim'
             'random'
             'fast_leaf_influence'
             'tree_loss'
)
flip_frac_list=(0.1 0.2 0.3 0.4)

for method in ${method_list[@]}; do
    for flip_frac_list in ${flip_frac_list[@]}; do
        job_name="C_${dataset}_${model}_${method}_${flip_frac}"
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/cleaning_new/$job_name \
               --error=jobs/errors/cleaning_new/$job_name \
               jobs/cleaning_new/runner.sh $dataset $model \
               $n_estimators $max_depth $method $flip_frac
    done
done
