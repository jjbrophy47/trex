sbatch jobs/performance/template.sh churn svm_linear
sbatch jobs/performance/template.sh churn svm_rbf
sbatch jobs/performance/template.sh churn lr
sbatch jobs/performance/template.sh churn knn

sbatch jobs/performance/template.sh amazon svm_linear
sbatch jobs/performance/template.sh amazon svm_rbf
sbatch jobs/performance/template.sh amazon lr
sbatch jobs/performance/template.sh amazon knn

sbatch jobs/performance/template.sh adult svm_linear
sbatch jobs/performance/template.sh adult svm_rbf
sbatch jobs/performance/template.sh adult lr
sbatch jobs/performance/template.sh adult knn

sbatch jobs/performance/template.sh census svm_linear
sbatch jobs/performance/template.sh census svm_rbf
sbatch jobs/performance/template.sh census lr
sbatch jobs/performance/template.sh census knn
