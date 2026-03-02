#include "lab2_alt_2.h"
#include <iostream>
#include <string>
#include <format>
#include <math.h>
#include "mpi.h"
using namespace std;

template <typename T, size_t N>
string arrToStr(T(&arr)[N]) {
	string result = "";
	for (size_t i = 0; i < N; i++) result += to_string(arr[i]) + " ";
	return result;
}

template <typename T, size_t N>
void printArr(T(&arr)[N]) {
	cout << arrToStr(arr) << endl;
}

template <typename T, size_t R, size_t C>
string matrixToStr(T(&matrix)[R][C]) {
	string result = "";
	for (size_t i = 0; i < R; i++) {
		for (size_t j = 0; j < C; j++) {
			result += to_string(matrix[i][j]) + " ";
		}
		result += "\n";
	}
	return result;
}

template <typename T, size_t R, size_t C>
void printMatrix(T(&matrix)[R][C]) {
	cout << matrixToStr(matrix) << endl;
}

void lab2_alt_2(int argc, char* argv[]) {
	const int
		COLS_NUM = 10,
		ROWS_NUM = 10,
		TOTAL_BLOCKS_NUM = 25,
		ELS_PER_BLOCK = 4,
		EVEN_BLOCKS_NUM = 12,
		ODD_BLOCKS_NUM = 13,
		TOTAL_PROCS_NUM = 24,
		EVEN_PROCS_NUM = 12,
		ODD_PROCS_NUM = 13;

	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Group
		worldGroup,
		evenGroup,
		oddGroup;

	int
		evenRanks[EVEN_PROCS_NUM],
		oddRanks[ODD_PROCS_NUM];

	evenRanks[0] = 0;
	oddRanks[0] = 0;
	for (size_t i = 1; i < TOTAL_PROCS_NUM; i++) {
		if (i % 2 == 0) evenRanks[i / 2] = i;
		else oddRanks[(i - 1) / 2 + 1] = i;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
	MPI_Group_incl(worldGroup, EVEN_PROCS_NUM, evenRanks, &evenGroup);
	MPI_Group_incl(worldGroup, ODD_PROCS_NUM, oddRanks, &oddGroup);

	MPI_Comm
		MPI_COMM_EVEN,
		MPI_COMM_ODD;

	MPI_Comm_create(MPI_COMM_WORLD, evenGroup, &MPI_COMM_EVEN);
	MPI_Comm_create(MPI_COMM_WORLD, oddGroup, &MPI_COMM_ODD);

	double blocks[TOTAL_BLOCKS_NUM][ELS_PER_BLOCK];
	double evenBlocks[EVEN_BLOCKS_NUM][ELS_PER_BLOCK];
	double oddBlocks[ODD_BLOCKS_NUM][ELS_PER_BLOCK];
	double block[ELS_PER_BLOCK];

	if (rank == 0) {
		double A[ROWS_NUM][COLS_NUM];
		double x = 0;
		for (size_t i = 0; i < ROWS_NUM; i++, x += 0.1) {
			double y = 0;
			for (size_t j = 0; j < COLS_NUM; j++, y += 0.1) {
				A[i][j] = sin(x) + exp(y);
			}
		}

		for (size_t i = 0; i < ROWS_NUM; i += 2) {
			for (size_t j = 0; j < COLS_NUM; j += 2) {
				int blockIdx = 5 * (i / 2) + (j / 2);
				blocks[blockIdx][0] = A[i][j];
				blocks[blockIdx][1] = A[i][j + 1];
				blocks[blockIdx][2] = A[i + 1][j];
				blocks[blockIdx][3] = A[i + 1][j + 1];
			}
		}

		for (size_t i = 0; i < TOTAL_BLOCKS_NUM; i++) {
			size_t blockIdx = i + 1;
			if (blockIdx % 2 == 0) {
				int evenBlockIdx = (i - 1) / 2;
				for (size_t j = 0; j < ELS_PER_BLOCK; j++) {
					evenBlocks[evenBlockIdx][j] = blocks[i][j];
				}
			}
			else {
				int oddBlockIdx = i / 2;
				for (size_t j = 0; j < ELS_PER_BLOCK; j++) {
					oddBlocks[oddBlockIdx][j] = blocks[i][j];
				}
			}
		}

		printMatrix(evenBlocks);
		printMatrix(oddBlocks);
	}

	if (MPI_COMM_EVEN != MPI_COMM_NULL) {
		int evenRank;
		MPI_Comm_rank(MPI_COMM_EVEN, &evenRank);

		MPI_Scatter(&evenBlocks[0], ELS_PER_BLOCK, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_EVEN);

		double
			localMin = INFINITY,
			globalMin = INFINITY;

		for (size_t i = 0; i < ELS_PER_BLOCK; i++) {
			if (localMin > block[i]) localMin = block[i];
		}
		cout << format("Локальный минимум для процесса {} группы EVEN: {}", evenRank, localMin) << endl;

		MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_EVEN);


		if (evenRank == 0) {
			cout << "Минимум группы EVEN: " << globalMin << endl;
		}
	}

	if (MPI_COMM_ODD != MPI_COMM_NULL) {
		int oddRank;
		MPI_Comm_rank(MPI_COMM_ODD, &oddRank);

		MPI_Scatter(&oddBlocks[0], ELS_PER_BLOCK, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_ODD);

		double
			localMax = -INFINITY,
			globalMax = -INFINITY;

		for (size_t i = 0; i < ELS_PER_BLOCK; i++) {
			if (localMax < block[i]) localMax = block[i];
		}

		cout << format("Локальный максимум для процесса {} группы ODD: {}", oddRank, localMax) << endl;

		MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_ODD);

		if (oddRank == 0) {
			cout << "Максимум группы ODD: " << globalMax << endl;
		}
	}

	if (MPI_COMM_EVEN != MPI_COMM_NULL) MPI_Comm_free(&MPI_COMM_EVEN);
	if (MPI_COMM_ODD != MPI_COMM_NULL) MPI_Comm_free(&MPI_COMM_ODD);
	if (evenGroup != MPI_GROUP_NULL) MPI_Group_free(&evenGroup);
	if (oddGroup != MPI_GROUP_NULL) MPI_Group_free(&oddGroup);
	if (worldGroup != MPI_GROUP_NULL) MPI_Group_free(&worldGroup);

	MPI_Finalize();
}