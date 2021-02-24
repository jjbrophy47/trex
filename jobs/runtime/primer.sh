dataset=$1
model=$2
n_estimators=$3
max_depth=$4
mem=$5
time=$6
partition=$7

rs_list=(1 2 3 4 5)
method_list=('klr-leaf_output' 'svm-leaf_output',
             'leaf_influence' 'maple' 'knn-leaf_output')

for method in ${method_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="RT_${dataset}_${method}"
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/runtime/$job_name \
               --error=jobs/errors/runtime/$job_name \
               jobs/runtime/runner.sh $dataset $model $n_estimators \
               $max_depth $method $rs
    done
done