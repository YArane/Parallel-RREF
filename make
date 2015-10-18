#!/bin/bash
mpicc.mpich2 cp_mpi.c -lm -o cp_mpi
mpiexec.mpich -np $1 ./cp_mpi $2
