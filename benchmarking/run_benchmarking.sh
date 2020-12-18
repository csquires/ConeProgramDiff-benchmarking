#!/usr/bin/env bash

rm -rf benchmarking/programs/
python3 -m benchmarking.save_benchmarking_programs
python3 -m benchmarking.benchmark_diffcp

