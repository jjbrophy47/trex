dataset=$1
mem=$2
time=$3
partition=$4

rs_list=(1 2 3 4 5)
model_list=('dt' 'lr' 'svm_linear' 'svm_rbf' 'knn')

for model in ${model_list[@]}; do
    for rs in ${rs_list[@]}; do
        job_name="P_${dataset}_${model}"

        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=$job_name \
               --output=jobs/logs/performance/$job_name \
               --error=jobs/errors/performance/$job_name \
               jobs/performance/runner.sh $dataset $model $rs
    done
done
