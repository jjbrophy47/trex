dataset=$1
mem=$2
time=$3
partition=$4

tune_frac=1.0
preprocessing='standard'
model_list=('cb' 'dt' 'lr' 'svm_linear' 'svm_rbf' 'knn')

for model in ${model_list[@]}; do
    for rs in {1..40}; do
        job_name="P_${dataset}_${model}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/performance/$job_name \
               --error=jobs/errors/performance/$job_name \
               jobs/performance/runner.sh $dataset $model $rs $preprocessing $tune_frac
    done
done
