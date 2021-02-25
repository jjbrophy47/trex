# all models
# ./jobs/performance/primer.sh 'churn' 2 1440 'short'
# ./jobs/performance/primer.sh 'surgical' 2 1440 'short'
# ./jobs/performance/primer.sh 'vaccine' 2 1440 'short'
# ./jobs/performance/primer.sh 'amazon' 20 1440 'short'
# ./jobs/performance/primer.sh 'bank_marketing' 2 1440 'short'
# ./jobs/performance/primer.sh 'adult' 2 1440 'short'
# ./jobs/performance/primer.sh 'census' 7 1440 'short'

# models that use a smaller validation set
# ./jobs/performance/primer_single.sh 'census' 'svm_rbf' 'standard' 0.25 20 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'svm_rbf' 'standard' 0.25 20 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'knn' 'standard' 0.25 10 1440 'short'

# RF
./jobs/performance/primer_single.sh 'churn' 'rf' 'standard' 1.0 2 1440 'short'
./jobs/performance/primer_single.sh 'surgical' 'rf' 'standard' 1.0 2 1440 'short'
./jobs/performance/primer_single.sh 'vaccine' 'rf' 'standard' 1.0 2 1440 'short'
./jobs/performance/primer_single.sh 'amazon' 'rf' 'standard' 1.0 10 1440 'short'
./jobs/performance/primer_single.sh 'bank_marketing' 'rf' 'standard' 1.0 2 1440 'short'
./jobs/performance/primer_single.sh 'adult' 'rf' 'standard' 1.0 2 1440 'short'
./jobs/performance/primer_single.sh 'census' 'rf' 'standard' 1.0 7 1440 'short'

# test ordinal categorical features for CatBoost
# ./jobs/performance/primer_single.sh 'churn' 'cb' 'categorical' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'surgical' 'cb' 'categorical' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'vaccine' 'cb' 'categorical' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'amazon' 'cb' 'categorical' 1.0 7 1440 'short'
# ./jobs/performance/primer_single.sh 'bank_marketing' 'cb' 'categorical' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'adult' 'cb' 'categorical' 1.0 2 1440 'short'
# ./jobs/performance/primer_single.sh 'census' 'cb' 'categorical' 1.0 7 1440 'short'
