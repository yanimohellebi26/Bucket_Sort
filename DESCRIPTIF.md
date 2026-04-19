# Bucket Sort with MPI — Distributed Sorting at Scale

## Description

A distributed implementation of the Bucket Sort algorithm using the MPI (Message Passing Interface) standard in C. The project also includes an optimised Top-K extraction variant, both benchmarked across different dataset sizes to measure real-world speedup when parallelising across multiple processes.

## What it brings

Shows how a classic sorting algorithm can be redesigned for distributed memory systems, with measurable performance gains. The benchmarks provide concrete evidence of the scalability benefits of parallelism — useful context for high-performance computing and data engineering roles.

## How it works

Each MPI process is assigned a bucket of the value range. The root process broadcasts the data, each worker sorts its own bucket locally, and the results are gathered back in order. The Top-K variant uses a similar pipeline but terminates early once the K largest values are confirmed. A Python plotting script visualises throughput across process counts.

## Status

✅ Complete — benchmarks run, CSV results exported, performance charts generated.

## Tech Stack

`C` · `MPI` · `OpenMPI` · `Python` · `Matplotlib` · `Bash`
