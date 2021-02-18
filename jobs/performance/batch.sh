# all models
# ./jobs/performance/primer.sh 'churn' 2 1440 'short'
./jobs/performance/primer.sh 'surgical' 5 1440 'short'
./jobs/performance/primer.sh 'vaccine' 5 1440 'short'
# ./jobs/performance/primer.sh 'amazon' 8 1440 'short'
./jobs/performance/primer.sh 'bank_marketing' 2 1440 'short'
# ./jobs/performance/primer.sh 'adult' 2 1440 'short'
# ./jobs/performance/primer.sh 'census' 5 1440 'short'

# single model
./jobs/performance/primer_single.sh 'census' lr 'standard' 20 1440 'short'
./jobs/performance/primer_single.sh 'census' svm_linear 'standard' 20 1440 'short'
./jobs/performance/primer_single.sh 'amazon' lr 'standard' 20 1440 'short'
./jobs/performance/primer_single.sh 'amazon' svm_linear 'standard' 20 1440 'short'

./jobs/performance/primer_single.sh 'churn' cb 'categorical' 2 1440 'short'
./jobs/performance/primer_single.sh 'surgical' cb 'categorical' 2 1440 'short'
./jobs/performance/primer_single.sh 'vaccine' cb 'categorical' 2 1440 'short'
./jobs/performance/primer_single.sh 'amazon' cb 'categorical' 8 1440 'short'
./jobs/performance/primer_single.sh 'bank_marketing' cb 'categorical' 2 1440 'short'
./jobs/performance/primer_single.sh 'adult' cb 'categorical' 2 1440 'short'
./jobs/performance/primer_single.sh 'census' cb 'categorical' 5 1440 'short'
