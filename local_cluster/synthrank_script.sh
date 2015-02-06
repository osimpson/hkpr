#!/bin/sh

python synth_rank_comparison.py pref 100 5 0.1 ./compare_rank/pref_100_5_0.1.txt
python synth_rank_comparison.py powercluster 100 5 0.1 ./compare_rank/powercluster_100_5_0.1.txt
python synth_rank_comparison.py smallworld 500 5 0.1 ./compare_rank/smallworld_500_5_0.1.txt
python synth_rank_comparison.py pref 500 5 0.1 ./compare_rank/pref_500_5_0.1.txt
python synth_rank_comparison.py smallworld 1000 5 0.1 ./compare_rank/smallworld_1000_5_0.1.txt
python synth_rank_comparison.py pref 1000 5 0.1 ./compare_rank/pref_1000_5_0.1.txt
python synth_rank_comparison.py powercluster 1000 5 0.1 ./compare_rank/powercluster_1000_5_0.1.txt
