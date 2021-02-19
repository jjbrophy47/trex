# all models and all tree kernels
./jobs/fidelity/primer.sh 'churn' 'cb' 100 5 2 1440 'short'
./jobs/fidelity/primer.sh 'surgical' 'cb' 100 5 2 1440 'short'
./jobs/fidelity/primer.sh 'vaccine' 'cb' 100 5 2 1440 'short'
./jobs/fidelity/primer.sh 'amazon' 'cb' 100 5 7 1440 'short'
./jobs/fidelity/primer.sh 'bank_marketing' 'cb' 100 5 2 1440 'short'
./jobs/fidelity/primer.sh 'adult' 'cb' 100 5 2 1440 'short'
./jobs/fidelity/primer.sh 'census' 'cb' 100 5 7 1440 'short'

# single settings
./jobs/fidelity/primer.sh 'census' 'cb' 100 5 'klr' 'leaf_output' 7 1440 'short'

