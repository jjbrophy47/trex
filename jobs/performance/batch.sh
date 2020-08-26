sbatch jobs/performance/runner.sh churn svm_linear
sbatch jobs/performance/runner.sh churn svm_rbf
sbatch jobs/performance/runner.sh churn lr
sbatch jobs/performance/runner.sh churn knn

sbatch jobs/performance/runner.sh amazon svm_linear
sbatch jobs/performance/runner.sh amazon svm_rbf
sbatch jobs/performance/runner.sh amazon lr
sbatch jobs/performance/runner.sh amazon knn

sbatch jobs/performance/runner.sh adult svm_linear
sbatch jobs/performance/runner.sh adult svm_rbf
sbatch jobs/performance/runner.sh adult lr
sbatch jobs/performance/runner.sh adult knn

sbatch jobs/performance/runner.sh census svm_linear
sbatch jobs/performance/runner.sh census svm_rbf
sbatch jobs/performance/runner.sh census svm_linear
sbatch jobs/performance/runner.sh census knn
