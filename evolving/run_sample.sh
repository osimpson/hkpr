#!/usr/bin/env bash
#edgelist=~/LLNL/local_code/git/hkpr/evolving/data/email-Eu-core-temporal.txt
#resultsdir=~/LLNL/local_code/git/hkpr/evolving/results
#logdir=~/LLNL/local_code/logs
edgelist=~/local_code/evolving/data/email-Eu-core-temporal.txt
resultsdir=~/local_code/evolving/results
logdir=~/local_code/logs

echo Running sampling process...

python sample.py --edgelist ${edgelist} --resultsdir ${resultsdir} &> ${logdir}/sample.log
