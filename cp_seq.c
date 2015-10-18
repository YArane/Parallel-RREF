#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "math.h"

#define EPSILON 0.000001

double ** allocate_matrix(int, int);
double ** read_user_matrix_from_file(char*, int*, int*);
void input_clicking_probabilities(double **, int, int, double *);
void write_clicking_probabilities_to_file(double *, int);
void print_matrix(double **, int, int);
void free_matrix(double **, int);
int equals(double, double);
void RREF(double **, int, int);
void divide_by_max(double **, int, int);
void print_best_acceptance_threshold(double *, int);


int main(int argc, char * argv[])
{
	struct timeval start, end; //struct used to compute execution time
	/* setup */
	int rows, columns;
	if (argc != 2) printf("please provide a user matrix!");
	gettimeofday(&start, NULL);
	double **A = read_user_matrix_from_file(argv[1], &rows, &columns);
	double *cp = malloc(columns * sizeof(double)); // clicking probabilities

	/* computation */
	RREF(A, rows, columns);
	divide_by_max(A, rows, columns);

	/* results */
	input_clicking_probabilities(A, rows, columns, cp);
	print_best_acceptance_threshold(cp, rows);
	write_clicking_probabilities_to_file(cp, rows);
	free_matrix(A, rows);
	free(cp);
		gettimeofday(&end, NULL);
		printf("\n\nAlgorithm's computational part duration :%ld\n", \
					((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
	return 0;
}

void input_clicking_probabilities(double **matrix, int rows, int columns, double *cp) {
	int row;
	for (row = 0; row < rows; row++) {
		cp[row] = matrix[row][columns-1];
	}
}

void write_clicking_probabilities_to_file(double *cp, int rows) {
	/* write clicking probabilities to file */ 
	FILE *output_file;
	int row;
	output_file = fopen("clicking_probabilities.txt","w");
	for (row = 0; row < rows; row++) {
		fprintf(output_file, "%lf\n", cp[row]);
	}

	fclose(output_file);
}

void RREF(double **matrix, int rows, int columns) {
	/* Gaussian elimination */
	int src_row, dest_row, row, row2, column;
	double pivot;
	for (src_row = 0; src_row < rows; src_row++) {
		for (dest_row = 0; dest_row < rows; dest_row++) {
			if (dest_row == src_row) continue;

			pivot = matrix[dest_row][src_row] / matrix[src_row][src_row];

			for (column = src_row; column < columns; column++) {
				matrix[dest_row][column] = matrix[dest_row][column] - pivot*matrix[src_row][column];
			}
		}
	}

	/* Back-substitution */
	for (row = rows-1; row >= 0; row--) {
		matrix[row][columns-1] = matrix[row][columns-1] / matrix[row][row];
		matrix[row][row] = 1;
		for (row2 = row-1; row2 >= 0; row2--) {
			matrix[row2][columns-1] += matrix[row2][row]*matrix[row][columns-1];
			matrix[row2][row] = 0;
		}
	}
}

void divide_by_max(double **matrix, int rows, int columns) {
	double max = 0; 
	int row, column;

	/* get max so we can divide by this later to get probabilities */
	for (row = 0; row < rows; row++) {		
		if (max < fabs(matrix[row][columns-1])) max = fabs(matrix[row][columns-1]);
	}

	/* divide by max and take abs */
	for (row = 0; row < rows; row++) {
		/* check for division by zero */
		if (equals(max,0)) {
			matrix[row][columns-1] = 0;
		} else {
			matrix[row][columns-1] = fabs (matrix[row][columns-1]) / max;
		}
	}
}

int equals(double a, double b) {
	if (fabs(a-b) < EPSILON) return 1;
	else return 0;
}

void print_matrix(double **matrix, int rows, int columns) 
{
	int row, column;
	for (row = 0; row < rows; row++) {
	 	for (column = 0; column < columns; column++) {
	 		printf("%lf ",matrix[row][column]);
	 	}
	 	printf("\n");
	}	
}

double ** allocate_matrix(int rows, int cols)
{
	int i = 0;
	double ** matrix = (double **) malloc(rows * sizeof(double *));

	for (i = 0; i < rows; i++) {
		matrix[i] = (double *) malloc(cols * sizeof(double));
	}

	return matrix;
}

double ** read_user_matrix_from_file(char *filename, int *rows, int *columns) {
	FILE *file;
	file = fopen(filename, "r");

	/* get number of rows and columns*/
	*rows = 1;
	*columns = 1;
	char c;
	int columns_known = 0;
	while(!feof(file)) {
		c = fgetc(file);
		if (c == ' ') {
			if (!columns_known) (*columns)++;
		} 

		if (c == '\n') {
			(*rows)++;
			columns_known = 1;
			continue;
		}
	}

	/* read values into array */
	rewind(file);
	int i, j;
	double **matrix = allocate_matrix(*rows, *columns);
	for (i = 0; i < *rows; i++) {
			for (j = 0; j < *columns; j++) {
					fscanf(file,"%lf",&matrix[i][j]);
		}
	} 
  	fclose(file);

	return matrix;
}

void free_matrix(double ** matrix, int rows)
{
	int i;
	for (i = 0; i < rows; i++) free(matrix[i]);
	free(matrix);
}

void print_best_acceptance_threshold(double *cp, int rows) {
/* TODO! (you didn't think this would ALL be handed to you, did you?) */
}
