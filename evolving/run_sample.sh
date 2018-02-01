#!/usr/bin/env bash

# set the code dirs
#edgelist=~/LLNL/local_code/git/hkpr/evolving/data/email-Eu-core-temporal.txt
#resultsdir=~/LLNL/local_code/git/hkpr/evolving/results
### BE CAREFUL NOT TO OVERWRITE RESULTS!!!
logdir=~/LLNL/local_code/logs
edgelist=~/local_code/evolving/data/email-Eu-core-temporal.txt
resultsdir=~/local_code/evolving/results
### BE CAREFUL NOT TO OVERWRITE RESULTS!!!
#logdir=~/local_code/logs

echo Running sampling process...

# if you want to see console printout, run this way
python sample.py --edgelist ${edgelist} --resultsdir ${resultsdir} --seednodes 90 --eps 0.1
# &> ${logdir}/sample.log

#python sample.py --edgelist ${edgelist} --resultsdir ${resultsdir} --seednodes 90 --eps 0.1 &> ${logdir}/sample.log