dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
mem=$6
time=$7
partition=$8

method_list=('random' 'klr-leaf_output' 'svm-leaf_output' 'knn-leaf_output' 'maple' 'leaf_influence')

for method in ${method_list[@]}; do
    for rs in {1..20}; do
        job_name="I_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/impact/$job_name \
               --error=jobs/errors/impact/$job_name \
               jobs/impact/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $rs
    done
done