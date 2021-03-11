# CatBoost: all surrogates, tree kernels, and metrics
./jobs/fidelity/primer.sh 'churn' 'cb' 'standard' 100 3 2 300 'short'
./jobs/fidelity/primer.sh 'surgical' 'cb' 'standard' 250 5 2 300 'short'
./jobs/fidelity/primer.sh 'vaccine' 'cb' 'standard' 250 5 10 300 'short'
./jobs/fidelity/primer.sh 'amazon' 'cb' 'standard' 250 7 20 300 'short'
./jobs/fidelity/primer.sh 'bank_marketing' 'cb' 'standard' 250 5 10 300 'short'
./jobs/fidelity/primer.sh 'adult' 'cb' 'standard' 250 5 10 300 'short'
./jobs/fidelity/primer.sh 'census' 'cb' 'standard' 250 5 30 300 'short'

# RF: all surrogates, tree kernels, and metrics
./jobs/fidelity/primer.sh 'churn' 'rf' 'standard' 100 7 2 300 'short'
./jobs/fidelity/primer.sh 'surgical' 'rf' 'standard' 250 7 20 300 'short'
./jobs/fidelity/primer.sh 'vaccine' 'rf' 'standard' 250 7 20 300 'short'
./jobs/fidelity/primer.sh 'amazon' 'rf' 'standard' 10 3 20 300 'short'
./jobs/fidelity/primer.sh 'bank_marketing' 'rf' 'standard' 250 7 20 300 'short'
./jobs/fidelity/primer.sh 'adult' 'rf' 'standard' 250 7 20 300 'short'
./jobs/fidelity/primer.sh 'census' 'rf' 'standard' 10 7 20 300 'short'
