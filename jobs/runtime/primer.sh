dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5 6 7 8 9 10)
method_list=('klr-leaf_output' 'svm-leaf_output' \
             'leaf_influence' 'maple' 'knn-leaf_output')

for method in ${method_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="RT_${dataset}_${model}_${method}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/runtime/$job_name \
               --error=jobs/errors/runtime/$job_name \
               jobs/runtime/runner.sh $dataset $model $preprocessing \
               $n_estimators $max_depth $method $rs
    done
done