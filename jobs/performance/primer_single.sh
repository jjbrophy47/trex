dataset=$1
model=$2
preprocessing=$3
tune_frac=$4
mem=$5
time=$6
partition=$7

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
