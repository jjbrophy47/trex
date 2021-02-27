# all surrogates, metrics, and tree kernels
./jobs/fidelity/primer.sh 'churn' 'cb' 'categorical' 100 3 2 1440 'short'
./jobs/fidelity/primer.sh 'surgical' 'cb' 'categorical' 250 5 2 1440 'short'
./jobs/fidelity/primer.sh 'vaccine' 'cb' 'categorical' 250 5 2 1440 'short'
./jobs/fidelity/primer.sh 'amazon' 'cb' 'categorical' 250 7 7 1440 'short'
./jobs/fidelity/primer.sh 'bank_marketing' 'cb' 'categorical' 250 3 2 1440 'short'
./jobs/fidelity/primer.sh 'adult' 'cb' 'categorical' 250 5 5 1440 'short'
./jobs/fidelity/primer.sh 'census' 'cb' 'categorical' 250 5 6 1440 'short'

# settings that require more memory
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'klr' 'leaf_output' 'mse' 21 1440 'short'
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'svm' 'leaf_output' 'mse' 21 1440 'short'
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'knn' 'leaf_output' 'mse' 21 1440 'short'
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'klr' 'leaf_path' 'mse' 30 1440 'short'
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'svm' 'leaf_path' 'mse' 30 1440 'short'
# ./jobs/fidelity/primer_single.sh 'census' 'cb' 'categorical' 250 5 'knn' 'leaf_path' 'mse' 30 1440 'short'

