#include <iostream>
#include <math.h>
#include <string>
#include <format>
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

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int
		ROWS_NUM = 10,
		COLS_NUM = 10,
		WORLD_GROUP_PROCS_NUM = 26,
		EVEN_GROUP_PROCS_NUM = 13,
		ODD_GROUP_PROCS_NUM = 14,
		EVEN_BLOCKS_NUM = 12,
		ODD_BLOCKS_NUM = 13,
		ELS_PER_BLOCK = 4;

	MPI_Group
		worldGroup,
		evenGroup,
		oddGroup;

	int
		evenRanks[EVEN_GROUP_PROCS_NUM],
		oddRanks[ODD_GROUP_PROCS_NUM];

	evenRanks[0] = 0;
	oddRanks[0] = 0;

	for (size_t i = 1; i < WORLD_GROUP_PROCS_NUM; i++) {
		if (i % 2 == 0) evenRanks[i / 2] = i;
		else oddRanks[(i - 1) / 2 + 1] = i;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
	MPI_Group_incl(worldGroup, EVEN_GROUP_PROCS_NUM, evenRanks, &evenGroup);
	MPI_Group_incl(worldGroup, ODD_GROUP_PROCS_NUM, oddRanks, &oddGroup);


	MPI_Comm
		MPI_COMM_EVEN,
		MPI_COMM_ODD;
	MPI_Comm_create(MPI_COMM_WORLD, evenGroup, &MPI_COMM_EVEN);
	MPI_Comm_create(MPI_COMM_WORLD, oddGroup, &MPI_COMM_ODD);


	double
		evenBlocks[EVEN_BLOCKS_NUM][ELS_PER_BLOCK],
		oddBlocks[ODD_BLOCKS_NUM][ELS_PER_BLOCK],
		block[ELS_PER_BLOCK];

	int sendCountsEven[EVEN_GROUP_PROCS_NUM],
		sendCountsOdd[ODD_GROUP_PROCS_NUM],
		displsEven[EVEN_GROUP_PROCS_NUM],
		displsOdd[ODD_GROUP_PROCS_NUM];

	if (rank == 0) {
		double matrix[ROWS_NUM][COLS_NUM];
		double x = 0;
		for (size_t i = 0; i < ROWS_NUM; i++, x += 0.1) {
			double y = 0;
			for (size_t j = 0; j < COLS_NUM; j++, y += 0.1) {
				matrix[i][j] = sin(x) + exp(y);
			}
		}

		for (size_t i = 0; i < ROWS_NUM; i += 2) {
			for (size_t j = 0; j < COLS_NUM; j += 2) {
				size_t blockIdx = (i / 2) * 5 + (j / 2) + 1;
				if (blockIdx % 2 == 0) {
					size_t evenBlockIdx = blockIdx / 2 - 1;
					evenBlocks[evenBlockIdx][0] = matrix[i][j];
					evenBlocks[evenBlockIdx][1] = matrix[i][j + 1];
					evenBlocks[evenBlockIdx][2] = matrix[i + 1][j];
					evenBlocks[evenBlockIdx][3] = matrix[i + 1][j + 1];
				}
				else {
					size_t oddBlockIdx = (blockIdx - 1) / 2;
					oddBlocks[oddBlockIdx][0] = matrix[i][j];
					oddBlocks[oddBlockIdx][1] = matrix[i][j + 1];
					oddBlocks[oddBlockIdx][2] = matrix[i + 1][j];
					oddBlocks[oddBlockIdx][3] = matrix[i + 1][j + 1];
				}
			}
		}

		for (size_t i = 0; i < EVEN_GROUP_PROCS_NUM; i++) {
			if (i == 0) sendCountsEven[i] = 0;
			else sendCountsEven[i] = ELS_PER_BLOCK;
		}

		for (size_t i = 0; i < ODD_GROUP_PROCS_NUM; i++) {
			if (i == 0) sendCountsOdd[i] = 0;
			else sendCountsOdd[i] = ELS_PER_BLOCK;
		}

		int displEven = 0;
		for (size_t i = 0; i < EVEN_GROUP_PROCS_NUM; i++) {
			if (i == 0) displsEven[i] = 0;
			else {
				displsEven[i] = displEven;
				displEven += ELS_PER_BLOCK;
			}
		}

		int displOdd = 0;
		for (size_t i = 0; i < ODD_GROUP_PROCS_NUM; i++) {
			if (i == 0) displsOdd[i] = 0;
			else {
				displsOdd[i] = displOdd;
				displOdd += ELS_PER_BLOCK;
			}
		}

		printMatrix(matrix);
		printMatrix(evenBlocks);
		printMatrix(oddBlocks);
		printArr(sendCountsEven);
		printArr(sendCountsOdd);
		printArr(displsEven);
		printArr(displsOdd);
	}

	double
		localMin = INFINITY,
		globalMin = INFINITY,
		localMax = -INFINITY,
		globalMax = -INFINITY;

	if (MPI_COMM_EVEN != MPI_COMM_NULL) {
		int evenRank;
		MPI_Comm_rank(MPI_COMM_EVEN, &evenRank);

		MPI_Scatterv(&evenBlocks[0], sendCountsEven, displsEven, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_EVEN);

		if (evenRank == 0) {
			MPI_Reduce(MPI_IN_PLACE, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_EVEN);
			cout << format("Глобальный минимум группы EVEN: {}", globalMin) << endl;
		}
		else {
			cout << format("Процесс с рангом {} группы EVEN получил блок: \n{}", evenRank, arrToStr(block)) << endl;

			localMin = block[0];
			for (size_t i = 1; i < ELS_PER_BLOCK; i++) {
				if (localMin > block[i]) localMin = block[i];
			}
			MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_EVEN);
		}

	}

	if (MPI_COMM_ODD != MPI_COMM_NULL) {
		int oddRank;
		MPI_Comm_rank(MPI_COMM_ODD, &oddRank);

		MPI_Scatterv(&oddBlocks[0], sendCountsOdd, displsOdd, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_ODD);

		if (oddRank == 0) {
			MPI_Reduce(MPI_IN_PLACE, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_ODD);
			cout << format("Глобальный максимум группы ODD: {}", globalMax) << endl;
		}
		else {
			cout << format("Процесс с рангом {} группы ODD получил блок: \n{}", oddRank, arrToStr(block)) << endl;

			localMax = block[0];
			for (size_t i = 1; i < ELS_PER_BLOCK; i++) {
				if (localMax < block[i]) localMax = block[i];
			}
			MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_ODD);
		}
	}

	MPI_Finalize();
	return 0;
}