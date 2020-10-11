/* DESCRIPTION: Conway's Game of Life project for UoA dit, September 2020.
 *
 * AUTHOR:      Andrew Pappas, 1115201500201, UoA, dit.
 * DATE:        September 2020.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "times.h"
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 32

// set FILE_INPUT for random input
#define FILE_INPUT 0

//If output is to be written in file set FILE_OUTPUT to 1 and set the correct output file name,
//otherwise set FILE_OUTPUT to write nothing
#define FILENAMEOUT "myouput.txt"
#define FILE_OUTPUT 1

//If starting or final board is to be printed set the flags to 1 from 0 respectively
#define PRINT_STARTING_BOARD 0
#define PRINT_FINAL_BOARD 0


//Copy real rows to "ghost" rows to make thε array cyclic
__global__ void copyRows(int *board, int dimension)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (id <= dimension)
	{
		//Copy first real row to bottom ghost row
		board[(dimension+2)*(dimension+1)+id] = board[(dimension+2)+id];
		//Copy last real row to top ghost row
		board[id] = board[(dimension+2)*dimension + id];
	}
}

//Copy real columns to "ghost" columns to make thε array cyclic
__global__ void copyColumns(int *board, int dimension)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id <= dimension+1)
	{
		//Copy first real column to right most ghost column
		board[id*(dimension+2)+dimension+1] = board[id*(dimension+2)+1];
		//Copy last real column to left most ghost column
		board[id*(dimension+2)] = board[id*(dimension+2) + dimension];
	}
}

//Get the game in the next generation
__global__ void nextGen(int *board, int *newBoard, int dimension)
{
	int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int id = iy * (dimension+2) + ix;

	int neighbors;

	if (iy <= dimension && ix <= dimension) {

	//count neighbors
	neighbors = board[id+(dimension+2)] + board[id-(dimension+2)] + board[id+1] + board[id-1] + board[id+(dimension+3)] + board[id-(dimension+3)] + board[id-(dimension+1)] + board[id+(dimension+1)];

	int value = board[id];
	//Apply the game's rules
	if (value == 1 && neighbors < 2)
		newBoard[id] = 0;
	else if (value == 1 && (neighbors == 2 || neighbors == 3))
		newBoard[id] = 1;
	else if (value == 1 && neighbors > 3)
		newBoard[id] = 0;
	else if (value == 0 && neighbors == 3)
		newBoard[id] = 1;
	else
		newBoard[id] = value;
	}
}

int main(int argc, char* argv[])
{
	int generations, dimension;
	int *board, *gpuBoard, *gpuNewBoard, *gpuTempBoard;
	FILE *fpOUT;
	//The boards that are going to be used are on-dimension boards even thought the game is
	//played in 2d boards. This is happening to make even faster and easier the coding. The formula
	//which is used to present virtually that 1d board as a 2d was found on the internet as a mathematic formula.

	if(argc!=3){
		printf("Correct usage of program is: %s dimensions generations\n",argv[0]);
		exit(-1);
	}

	dimension = atoi(argv[1]);
	generations = atoi(argv[2]);



	printf("Running Conway's Game of life with:\n");
	printf("Dimension = %d\n",dimension);
	printf("Generations = %d\n",generations);

	size_t size = sizeof(int)*(dimension+2)*(dimension+2);

	//Allocate ram memory equal to the size of the board with the "ghost" cells
	board = (int*)malloc(size);

	if(board == NULL){
		printf("Error in ram memory allocation\n");
		exit(-1);
	}

	srand(0);
	int i,j;

	#if FILE_INPUT==0

	//Initialization of the starting board randomly
	for(i = 1; i<=dimension; i++) {
		for(j = 1; j<=dimension; j++) {
			board[i*(dimension+2)+j] = rand() % 2;
		}
	}

	#endif

	#if PRINT_STARTING_BOARD==1

	printf("\nPrinting starting board:\n");
	for(i = 1; i<=dimension; i++) {
		for(j = 1; j<=dimension; j++) {
			printf("%d ", board[i*(dimension+2)+j]);
		}
		printf("\n");
	}
	printf("\n");

	#endif

	//Allocate gpu memory equal to the size of the board with the "ghost" cells twice
	cudaMalloc(&gpuBoard, size);//one for the "old" board in each generation transmission
	cudaMalloc(&gpuNewBoard, size);//and one for the "new" one

	//Copying the starting board from the ram to the gpu memory
	cudaMemcpy(gpuBoard, board, size, cudaMemcpyHostToDevice);

	//Defining the grid of blocks and threads
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
	int linGrid = (int)ceil(dimension/(float)BLOCK_SIZE);
	dim3 boardSize(linGrid,linGrid,1);
	dim3 cblock_size(BLOCK_SIZE,1,1);
	dim3 rows_size((int)ceil(dimension/(float)cblock_size.x),1,1);
	dim3 columns_size((int)ceil((dimension+2)/(float)cblock_size.x),1,1);

 	//Starting the timer
	timestamp start = getTime();

	int k;
	//Loop for every generation
	for(k=0; k<generations; k++){
		//Creating the "ghost" cells by copying the perimeter cells (rows and columns) to them
		copyRows<<<rows_size, cblock_size>>>(gpuBoard, dimension);
		copyColumns<<<columns_size, cblock_size>>>(gpuBoard, dimension);

		//Proceeding to the next generation of the game saving it to the gpuNewBoard
		nextGen<<<boardSize, blockSize>>>(gpuBoard, gpuNewBoard, dimension);

		//Swaping the old with the new board to avoid re-allocing space
		gpuTempBoard = gpuBoard;
		gpuBoard = gpuNewBoard;
		gpuNewBoard = gpuTempBoard;
	}

 	//Copying the final board the gpu memory to ram
	cudaMemcpy(board, gpuBoard, size, cudaMemcpyDeviceToHost);

	//Finishing the timer
	float elapsedTime = getElapsedTime(start);

	int sum = 0;
	//Adding up the alive
	for (i = 1; i<=dimension; i++) {
		for (j = 1; j<=dimension; j++) {
			sum += board[i*(dimension+2)+j];
		}
	}

	printf("Total game time: %.2f msecs\n", elapsedTime);
	printf("Total finally alive: %d\n", sum);

	#if PRINT_FINAL_BOARD==1

	printf("\nPrinting final board:\n");
	for(i = 1; i<=dimension; i++) {
		for(j = 1; j<=dimension; j++) {
			printf("%d ", board[i*(dimension+2)+j]);
		}
		printf("\n");
	}
	printf("\n");

	#endif

	#if FILE_OUTPUT==1

	fpOUT = fopen(FILENAMEOUT,"w");
	if (fpOUT == NULL)
	{
		printf("Output file didn't open properly.\n");
		exit(-1);
	}

	for(i = 1; i<=dimension; i++) {
		for(j = 1; j<=dimension; j++) {
			fprintf(fpOUT, "%d ", board[i*(dimension+2)+j]);
		}
		fprintf(fpOUT, "\n");
	}

	fclose(fpOUT);

	#endif

	//Deleting the allocated gpu memory
	cudaFree(gpuBoard);
	cudaFree(gpuNewBoard);

	//Deleting the allocated ram memory
	free(board);

	return 0;
}
