# TREX - CB
# ./jobs/cleaning/trex_primer.sh churn 100 3 0.3 1.0 3 1440 short
# ./jobs/cleaning/trex_primer.sh amazon 250 5 0.15 1.0 5 1440 short
# ./jobs/cleaning/trex_primer.sh adult 100 5 0.25 1.0 6 1440 short
# ./jobs/cleaning/trex_primer.sh census 250 5 0.15 0.1 15 1440 short
# ./jobs/cleaning/trex_primer.sh census 250 5 0.15 1.0 30 1440 short

# TREX - RF
./jobs/cleaning/trex_primer.sh churn rf 250 10 0.3 1.0 6 1440 short
./jobs/cleaning/trex_primer.sh amazon rf 100 10 0.15 1.0 7 1440 short
./jobs/cleaning/trex_primer.sh adult rf 250 10 0.25 1.0 7 1440 short
./jobs/cleaning/trex_primer.sh census rf 250 5 0.15 1.0 30 1440 short

# TEKNN - CB
# ./jobs/cleaning/teknn_primer.sh churn 100 3 0.3 1.0 3 1440 short
# ./jobs/cleaning/teknn_primer.sh amazon 250 5 0.15 1.0 5 1440 short
# ./jobs/cleaning/teknn_primer.sh adult 100 5 0.25 1.0 6 1440 short
# ./jobs/cleaning/teknn_primer.sh census 250 5 0.15 0.1 15 1440 short
# ./jobs/cleaning/teknn_primer.sh census 250 5 0.15 1.0 30 1440 short

# TEKNN - RF
./jobs/cleaning/teknn_primer.sh churn rf 250 10 0.3 1.0 6 1440 short
./jobs/cleaning/teknn_primer.sh amazon rf 100 10 0.15 1.0 7 1440 short
./jobs/cleaning/teknn_primer.sh adult rf 250 10 0.25 1.0 7 1440 short
./jobs/cleaning/teknn_primer.sh census rf 250 5 0.15 1.0 30 1440 short

# MAPLE - CB
# ./jobs/cleaning/maple_primer.sh churn 100 3 0.3 1.0 3 1440 short
# ./jobs/cleaning/maple_primer.sh amazon 250 5 0.15 1.0 5 1440 short
# ./jobs/cleaning/maple_primer.sh adult 100 5 0.25 1.0 6 1440 short
# ./jobs/cleaning/maple_primer.sh census 250 5 0.15 0.1 15 1440 short
# ./jobs/cleaning/maple_primer.sh census 250 5 0.15 1.0 20 1440 short

# MAPLE - RF
./jobs/cleaning/maple_primer.sh churn rf 250 10 0.3 1.0 6 1440 short
./jobs/cleaning/maple_primer.sh amazon rf 100 10 0.15 1.0 7 1440 short
./jobs/cleaning/maple_primer.sh adult rf 250 10 0.25 1.0 7 1440 short
./jobs/cleaning/maple_primer.sh census rf 250 5 0.15 1.0 20 1440 short

# Leaf Influence
# ./jobs/cleaning/influence_primer.sh churn 100 3 0.3 1.0 3 1440 short
# ./jobs/cleaning/influence_primer.sh amazon 250 5 0.15 1.0 5 1440 short
# ./jobs/cleaning/influence_primer.sh adult 100 5 0.25 1.0 6 1440 short
# ./jobs/cleaning/influence_primer.sh census 250 5 0.15 0.1 15 1440 short
# ./jobs/cleaning/influence_primer.sh census 250 5 0.15 1.0 20 1440 short

# # MMD-Critic
# ./jobs/cleaning/mmd_primer.sh churn 100 3 0.3 1.0 3 1440 short
# ./jobs/cleaning/mmd_primer.sh amazon 250 5 0.15 1.0 5 1440 short
# ./jobs/cleaning/mmd_primer.sh adult 100 5 0.25 1.0 6 1440 short
# ./jobs/cleaning/mmd_primer.sh census 250 5 0.15 0.1 15 1440 short
# ./jobs/cleaning/mmd_primer.sh census 250 5 0.15 1.0 30 1440 short

# TreeProto
./jobs/cleaning/proto_primer.sh churn rf 250 10 0.3 1.0 6 1440 short
./jobs/cleaning/proto_primer.sh amazon rf 100 10 0.15 1.0 7 1440 short
./jobs/cleaning/proto_primer.sh adult rf 250 10 0.25 1.0 7 1440 short
./jobs/cleaning/proto_primer.sh census rf 250 5 0.15 1.0 30 1440 short
