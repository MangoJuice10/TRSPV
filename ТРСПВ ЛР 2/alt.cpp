#include <iostream>
#include <math.h>
#include <string>
#include <sstream>
#include <format>
#include "mpi.h"
using namespace std;

template <typename T, size_t N>
string arrToStr(T(&arr)[N]) {
	string result = "";
	for (size_t i = 0; i < N; i++) {
		result += to_string(arr[i]) + " ";
	}
	return result;
}

template <typename T, size_t N>
void printArr(T(&arr)[N]) {
	cout << arrToStr(arr) << endl;
}

template <size_t R, size_t C>
string matrixToStr(double(&matrix)[R][C]) {
	string result = "";
	ostringstream oss;
	for (size_t i = 0; i < R; i++) {
		for (size_t j = 0; j < C; j++) {
			oss.str("");
			oss.clear();
			oss << round(matrix[i][j] * 100) / 100;
			result += oss.str() + " ";
		}
		result += "\n";
	}
	return result;
}

template <size_t R, size_t C>
void printMatrix(double(&matrix)[R][C]) {
	cout << matrixToStr(matrix) << endl;
}

int temp(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const size_t
		ROWS_NUM = 10,
		COLS_NUM = 10,

		BLOCKS_NUM = 25,
		ELS_PER_BLOCK = 4,

		TOTAL_PROCS_NUM = 49,
		GROUP_PROCS_NUM = 25;

	MPI_Group
		worldGroup,
		minGroup,
		maxGroup;
	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	int
		minRanks[GROUP_PROCS_NUM],
		maxRanks[GROUP_PROCS_NUM];

	minRanks[0] = 0;
	maxRanks[0] = 0;

	for (int i = 1; i < GROUP_PROCS_NUM; i++) {
		minRanks[i] = i;
		maxRanks[i] = 24 + i;
	}

	MPI_Group_incl(worldGroup, GROUP_PROCS_NUM, minRanks, &minGroup);
	MPI_Group_incl(worldGroup, GROUP_PROCS_NUM, maxRanks, &maxGroup);

	MPI_Comm
		MPI_COMM_MIN,
		MPI_COMM_MAX;

	MPI_Comm_create(MPI_COMM_WORLD, minGroup, &MPI_COMM_MIN);
	MPI_Comm_create(MPI_COMM_WORLD, maxGroup, &MPI_COMM_MAX);

	double blocks[BLOCKS_NUM][ELS_PER_BLOCK];
	double block[ELS_PER_BLOCK];

	if (rank == 0) {
		cout << format("Ранги процессов группы MIN: {}", arrToStr(minRanks)) << endl;
		cout << format("Ранги процессов группы MAX: {}", arrToStr(maxRanks)) << endl;

		double A[ROWS_NUM][COLS_NUM];

		double x = 0;
		for (size_t i = 0; i < ROWS_NUM; i++, x += 0.1) {
			double y = 0;
			for (size_t j = 0; j < COLS_NUM; j++, y += 0.1) {
				A[i][j] = sin(x) + exp(y);
			}
		}

		cout << endl;
		cout << format("Матрица вычисленных значений функции f(x): \n{}", matrixToStr(A)) << endl;

		for (size_t i = 0; i < ROWS_NUM; i += 2) {
			for (size_t j = 0; j < COLS_NUM; j += 2) {
				int blockIdx = (int)(5 * (i / 2) + j / 2);
				blocks[blockIdx][0] = A[i][j];
				blocks[blockIdx][1] = A[i][j + 1];
				blocks[blockIdx][2] = A[i + 1][j];
				blocks[blockIdx][3] = A[i + 1][j + 1];
			}
		}
		cout << endl;
		cout << format("Блоки для отправки в процессы: \n{}", matrixToStr(blocks)) << endl;
	}

	if (MPI_COMM_MIN != MPI_COMM_NULL) {
		double
			localMin,
			globalMin = INFINITY;

		int minGroupRank;
		MPI_Comm_rank(MPI_COMM_MIN, &minGroupRank);
		MPI_Scatter(&blocks[0], ELS_PER_BLOCK, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_MIN);

		cout << format("Процесс с рангом {} группы MIN получил блок: {}", minGroupRank, arrToStr(block)) << endl;
		localMin = block[0];
		for (int i = 1; i < ELS_PER_BLOCK; i++) {
			if (localMin > block[i]) localMin = block[i];
		}
		cout << format("Локальный минимум процесса с рангом {}: {}", minGroupRank, localMin) << endl;

		MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_MIN);
		if (minGroupRank == 0) cout << "Глобальный минимум: " << globalMin << endl << endl;
	}

	if (MPI_COMM_MAX != MPI_COMM_NULL) {
		double
			localMax,
			globalMax = -INFINITY;

		int maxGroupRank;
		MPI_Comm_rank(MPI_COMM_MAX, &maxGroupRank);
		MPI_Scatter(&blocks[0], ELS_PER_BLOCK, MPI_DOUBLE, &block[0], ELS_PER_BLOCK, MPI_DOUBLE, 0, MPI_COMM_MAX);

		cout << format("Процесс с рангом {} группы MAX получил блок: {}", maxGroupRank, arrToStr(block)) << endl;
		localMax = block[0];
		for (int i = 1; i < ELS_PER_BLOCK; i++) {
			if (localMax < block[i]) localMax = block[i];
		}
		cout << format("Локальный максимум процесса с рангом {}: {}", maxGroupRank, localMax) << endl;

		MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_MAX);
		
		if (maxGroupRank == 0) cout << "Глобальный максимум: " << globalMax << endl << endl;
	} 

	if (MPI_COMM_MIN != MPI_COMM_NULL) MPI_Comm_free(&MPI_COMM_MIN);
	if (MPI_COMM_MAX != MPI_COMM_NULL) MPI_Comm_free(&MPI_COMM_MAX);
	if (minGroup != MPI_GROUP_NULL) MPI_Group_free(&minGroup);
	if (maxGroup != MPI_GROUP_NULL) MPI_Group_free(&maxGroup);
	if (worldGroup != MPI_GROUP_NULL) MPI_Group_free(&worldGroup);
	MPI_Finalize();
	return 0;
}