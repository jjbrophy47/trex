# CatBoost: all methods, experiments are executed in chunks of 20 runs at a time
./jobs/roar/primer.sh 'churn' 'cb' 'categorical' 100 3 0.50 20 2 1440 'short' 41
./jobs/roar/primer.sh 'surgical' 'cb' 'categorical' 250 5 0.50 20 2 1440 'short' 41
./jobs/roar/primer.sh 'vaccine' 'cb' 'categorical' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'amazon' 'cb' 'categorical' 250 7 0.50 20 15 1440 'short' 41
./jobs/roar/primer.sh 'bank_marketing' 'cb' 'categorical' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'adult' 'cb' 'categorical' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'census' 'cb' 'categorical' 250 5 0.50 20 20 1440 'short' 41

# RF: all methods, experiments are executed in chunks of 20 runs at a time
./jobs/roar/primer.sh 'churn' 'rf' 'standard' 100 3 0.50 20 2 1440 'short' 41
./jobs/roar/primer.sh 'surgical' 'rf' 'standard' 250 5 0.50 20 2 1440 'short' 41
./jobs/roar/primer.sh 'vaccine' 'rf' 'standard' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'amazon' 'rf' 'categorical' 250 7 0.50 20 15 1440 'short' 41
./jobs/roar/primer.sh 'bank_marketing' 'rf' 'standard' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'adult' 'rf' 'standard' 250 5 0.50 20 10 1440 'short' 41
./jobs/roar/primer.sh 'census' 'rf' 'standard' 250 5 0.50 20 20 1440 'short' 41
