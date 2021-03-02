# all models
# ./jobs/performance/primer.sh 'churn' 2 1440 'short'
# ./jobs/performance/primer.sh 'surgical' 2 1440 'short'
# ./jobs/performance/primer.sh 'vaccine' 2 1440 'short'
# ./jobs/performance/primer.sh 'amazon' 20 1440 'short'
# ./jobs/performance/primer.sh 'bank_marketing' 2 1440 'short'
# ./jobs/performance/primer.sh 'adult' 2 1440 'short'
# ./jobs/performance/primer.sh 'census' 7 1440 'short'

# models that use a smaller validation set, need to cancel some from above
# scancel --name=P_census_svm_rbf
# scancel --name=P_amazon_svm_rbf
# scancel --name=P_amazon_knn
# ./jobs/performance/primer_single.sh 'census' 'svm_rbf' 'standard' 'accuracy' 0.25 20 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'svm_rbf' 'standard' 'accuracy' 0.25 20 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'knn' 'standard' 'accuracy' 0.25 10 1440 'short'

# CB using categorical
# ./jobs/performance/primer_single.sh 'churn' 'cb' 'categorical' 'accuracy' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'surgical' 'cb' 'categorical' 'accuracy' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'vaccine' 'cb' 'categorical' 'accuracy' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'cb' 'categorical' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'bank_marketing' 'cb' 'categorical' 'accuracy' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'adult' 'cb' 'categorical' 'accuracy' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'census' 'cb' 'categorical' 'accuracy' 1.0 7 1440 'short'

# RF
# ./jobs/performance/primer_single.sh 'churn' 'rf' 'standard' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'surgical' 'rf' 'standard' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'vaccine' 'rf' 'standard' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'rf' 'standard' 'accuracy' 1.0 15 1440 'short'
# ./jobs/performance/primer_single.sh 'bank_marketing' 'rf' 'standard' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'adult' 'rf' 'standard' 'accuracy' 1.0 10 1440 'short'
# ./jobs/performance/primer_single.sh 'census' 'rf' 'standard' 'accuracy' 1.0 15 1440 'short'

# RF scoring using AUC
./jobs/performance/primer_single.sh 'amazon' 'rf' 'standard' 'roc_auc' 1.0 15 1440 'short'