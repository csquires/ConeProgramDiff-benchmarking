#!/usr/bin/env bash

rm -rf benchmarking/programs/
python3 -m benchmarking.save_benchmarking_programs
python3 -m benchmarking.benchmark_diffcp
julia benchmarking/benchmark_cone_program_diff.jl
python3 -m benchmarking.plot_benchmarking_results

