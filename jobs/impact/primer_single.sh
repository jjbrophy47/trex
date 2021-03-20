dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
method=$6
start_pred=$7
setting=$8
$n_test=$9
mem=${10}
time=${11}
partition=${12}

for rs in {1..20}; do
    job_name="I_${dataset}_${model}_${method}_${setting}_${start_pred}_${n_test}"

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/impact/$job_name \
           --error=jobs/errors/impact/$job_name \
           jobs/impact/runner.sh $dataset $model $preprocessing \
           $n_estimators $max_depth $method $start_pred $setting $n_test $rs
done