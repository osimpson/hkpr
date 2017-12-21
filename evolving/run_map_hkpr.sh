#!/usr/bin/env bash
edgelist=~/LLNL/local_code/git/hkpr/evolving/data/email-Eu-core-temporal.txt
resultsdir=~/LLNL/local_code/git/hkpr/evolving/results
logdir=~/LLNL/local_code/logs

python map_hkpr.py --edgelist ${edgelist} --resultsdir ${resultsdir} &> ${logdir}/map_hkpr.log
