#!/usr/bin/env bash

echo SEED90
for f in exact/seed90/*.txt
do
  python draw_hkpr.py $f
done

echo SEED111
for f in exact/seed111/*.txt
do
  python draw_hkpr.py $f
done

echo SEED121
for f in exact/seed121/*.txt
do
  python draw_hkpr.py $f
done

echo SEED851
for f in exact/seed851/*.txt
do
  python draw_hkpr.py $f
done