#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void get_size_of_matrix(char *);
void get_lineSize(int *);
double * read_user_matrix(char *, int);
double * load_user_matrix();
void print_matrix(double *, int, int);
void print(double *);
void RREF(double *, int); 
void REF(double *, int, int);

int rank, np;
int rows, columns;
int bufferSize, lines;
double *m;
MPI_Status status;

int main(int argc, char *argv[]){
	char *fileName;
	int lineSize, charsPerLine;
	

	//check user input
	if(argc != 2){
		printf("usage: mpirun -np # ./cp_mpi matrix.txt.\n");
		return -1;
	}

	//initialize MPI
	if(MPI_SUCCESS != MPI_Init(&argc, &argv)){
		printf("error initializing MPI.\n");
		exit(-1);	
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	fileName = argv[1];
	
	//get size of matrix
	get_size_of_matrix(fileName);
	//get size of each line (in bytes)
	get_lineSize(&lineSize);
	//read user matrix
		/*
			note, to ensure that the matrix will always be located in a contiguous
			portion of memory, only a single dimensional array will be used to hold the 
			matrix. 
		*/
	double *matrix = read_user_matrix(fileName, lineSize);
//	print_matrix(matrix, lines, columns);
	//create clicking probabilities vector

	if((m = malloc((rows*columns) * sizeof(double))) == NULL){
			printf("error allocating matrix.\n");
			exit(-1);
	}	
	//computation//
	RREF_p(matrix, lines, columns);	
//	print_matrix(matrix, lines, columns);
//	print(m);
	RREF(m, rows);
	print(m);
	//terminate MPI
	if(MPI_SUCCESS != MPI_Finalize()){
		printf("error terminating MPI.\n");
		exit(-1);
	}

	return 0; 
}

void REF(double *matrix, int row_limit, int columns){
	
	int row, j, i;
	double *tmp;
	if((tmp = malloc(columns*row_limit*sizeof(double))) == NULL){
		printf("error mallocing.\n");
		exit(-1);
	}	

	for(i=0;i<rank*row_limit;i++){
		MPI_Bcast(tmp, columns, MPI_DOUBLE, i/row_limit, MPI_COMM_WORLD);
		//adjust rows based on tmp
		for(row=0;row<row_limit;row++){
			for(j=i+1;j<columns;j++){
				matrix[row*columns + j] = matrix[row*columns + j] - matrix[row*columns + i] * tmp[j];
			}	
			matrix[row*columns + i] = 0;
		}
	}

	for(row=0;row<row_limit;row++){
		for(j=rank*row_limit+row+1;j<columns;j++){
			matrix[row*columns+j] = matrix[row*columns+j] / matrix[row*columns+rank*row_limit+row];
		}
		matrix[row*columns+rank*row_limit+row] = 1;

		for(i=0;i<columns;i++){
			tmp[i] = matrix[row*columns+i];
		}
		MPI_Bcast(tmp, columns, MPI_DOUBLE, rank, MPI_COMM_WORLD);

		for(i=row+1;i<row_limit;i++){
			for(j=rank*row_limit+row+1;j<columns;j++){
				matrix[i*columns+j] = matrix[i*columns+j] - matrix[i*columns+row+rank*row_limit]*tmp[j];
			}
			matrix[i*columns+row+rank*row_limit] = 0;
		}
	}
	for(i=(rank+1)*row_limit;i<columns-1;i++){
		MPI_Bcast(tmp, columns, MPI_DOUBLE, i/row_limit, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(matrix,columns*row_limit,MPI_DOUBLE,m,columns*row_limit,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


void RREF(double *matrix, int rows){
	int row, row2;
	for(row=rows-1;row>=0;row--){
		matrix[row*columns+columns-1] = matrix[row*columns+columns-1]/matrix[row*columns+row];
		matrix[row*columns+row] = 1;
		for(row2=row-1;row2>=0;row2--){
			matrix[row2*columns+columns-1] += matrix[row2*columns+row] * matrix[row*columns+columns-1];
			matrix[row2*columns+row] = 0;
		}	
	}
}

double * allocate_matrix(int lines){
	double *matrix;
	if((matrix = malloc((lines*columns) * sizeof(double))) == NULL){
			printf("error allocating matrix.\n");
			exit(-1);
	}	
	return matrix;
}

void print(double *matrix){
	if(rank == 0){
	int i, j;
	for(i=0;i<rows;i++){
		for(j=0;j<columns;j++){
			printf("%lf ", matrix[i*columns+j]);
		}
		printf("\n");
	}
	}
}

void print_matrix(double *matrix, int i, int j) 
{
	if(rank == 0){
		int row, column;
		for (row = 0; row < i; row++) {
	 		for (column = 0; column < j; column++) {
	 			printf("%lf ",matrix[row*columns+column]);
		 	}
		 	printf("\n");
		}	

		//print from other process
		int p;
		for(p=1;p<np;p++){
			double *buffer = malloc(9*4*sizeof(double));
			if(buffer == NULL){
				printf("error mallocing.\n");
				exit(-1);
			}
			MPI_Recv(buffer, 9*4, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &status);
			for(row =0;row<i;row++){
				for (column = 0; column < j; column++) {
					printf("%lf ", buffer[row*columns + column]);
				}
				printf("\n");
			}
		}
	}else{
		MPI_Send(matrix, 9*4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}

//loads contents in the buffer into a matrix
double * load_user_matrix(char *buffer, int lineSize){
	lines = bufferSize / lineSize;
	
	double *matrix = allocate_matrix(lines);
	
	char temp[8];
	int i, j, k;
	j = i = k = 0;
	while(*buffer != '\0'){
		if(*buffer == '\n'){
			i++;
			j=0;
		}else if(*buffer == ' '){
			j++;
		}else{
			temp[k++] = *buffer;
		}
		if(k == 8){
			k = 0;
			matrix[i*columns+j] = atof(temp);
		}
		*buffer++;
	}
	return matrix;
}

//reads matrix from file and stores into buffers
double * read_user_matrix(char *fileName, int lineSize){
	MPI_File file;

	//every process will get a buffer to read its section
	//lets divide up our file..

	//how many lines per buffer?
	int linesPerBuffer = rows / np;
	//don't forget the overflow!
	int overflow = rows % np; //overflow: [0- np-1]

	//so what's the size of each buffer? 
		//overflow check
	bufferSize = (rank < overflow)? (linesPerBuffer+1) * lineSize : linesPerBuffer * lineSize; 


	//lets create the buffer
	char *buffer;
	if((buffer = malloc(bufferSize)) == NULL){
		printf("error creating buffer.\n");
	}

	//lastly, each process needs to know where to seek the file pointer
		//because of the overflow, it's not just a simple multiplication
	int seek;
	if(rank == 0){
		seek = 0;
	}else if(overflow == 0){
		seek = ((linesPerBuffer)*lineSize * rank);
	}else if(rank <= overflow){
		seek = ((linesPerBuffer+1)*lineSize * rank);
	}else{
		seek = ((linesPerBuffer+1)*lineSize * (rank-1)) + ((linesPerBuffer)*lineSize * (rank-overflow));
	}


	MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	MPI_File_seek(file, seek, MPI_SEEK_SET);
	MPI_File_read(file, buffer, bufferSize, MPI_CHAR, &status);


	MPI_Barrier(MPI_COMM_WORLD); //wait until all processes finished reading
	MPI_File_close(&file);

	return load_user_matrix(buffer, lineSize);
}

//determine size of each line in bytes
void get_lineSize(int *lineSize){
	//			each element in the row   +   the spaces       + the carriage return					 
	*lineSize = sizeof(double) * columns + sizeof(char) * (columns-1) + sizeof(char);
}

//determine size of matrix
void get_size_of_matrix(char *fileName){
	FILE *file;
	//open & check if valid file
	if((file = fopen(fileName, "r")) == 0){
		printf("error openening file %s.\n", fileName);
		exit(-1);
	}

	rows = 1;
	columns = 1;
	char c;
	int columns_known = 0;
	while(!feof(file)){
		c = fgetc(file);
		if(c == ' '){
			if(!columns_known)
				(columns)++;
		}
		if(c == '\n'){
			(rows)++;
			columns_known = 1;
			continue;
		}
	}
	fclose(file);
}
