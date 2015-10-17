program : cp_mpi.o 
		mpicc -o program cp_mpi.o 

cp_mpi.o : cp_mpi.c 
		mpicc -c cp_mpi.c

clean : 
		rm -f program cp_mpi.o 
