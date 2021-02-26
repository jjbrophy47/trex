dataset=$1
mem=$2
time=$3
partition=$4

tune_frac=1.0
processing='standard'
rs_list=(1 2 3 4 5 6 7 8 9 10)
model_list=('cb' 'dt' 'lr' 'svm_linear' 'svm_rbf' 'knn')

for model in ${model_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="P_${dataset}_${model}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/performance/$job_name \
               --error=jobs/errors/performance/$job_name \
               jobs/performance/runner.sh $dataset $model $rs $processing $tune_frac
    done
done
