./jobs/performance/primer.sh adult 5 1440 short
./jobs/performance/primer.sh amazon 5 1440 short
./jobs/performance/primer.sh census 5 1440 short
./jobs/performance/primer.sh churn 5 1440 short


# sbatch jobs/performance/primer.sh churn svm_linear
# sbatch jobs/performance/primer.sh churn svm_rbf
# sbatch jobs/performance/primer.sh churn lr
# sbatch jobs/performance/primer.sh churn knn

# sbatch jobs/performance/primer.sh amazon svm_linear
# sbatch jobs/performance/primer.sh amazon svm_rbf
# sbatch jobs/performance/primer.sh amazon lr
# sbatch jobs/performance/primer.sh amazon knn

# sbatch jobs/performance/primer.sh adult svm_linear
# sbatch jobs/performance/primer.sh adult svm_rbf
# sbatch jobs/performance/primer.sh adult lr
# sbatch jobs/performance/primer.sh adult knn

# sbatch jobs/performance/primer.sh census svm_linear
# sbatch jobs/performance/primer.sh census svm_rbf
# sbatch jobs/performance/primer.sh census svm_linear
# sbatch jobs/performance/primer.sh census knn
