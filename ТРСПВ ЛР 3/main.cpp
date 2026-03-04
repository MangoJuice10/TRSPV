#include <iostream>
#include <string>
#include <format>
#include "mpi.h"
using namespace std;

template <typename T, size_t N>
string arrToStr(T(&arr)[N]) {
	string result = "";
	for (int i = 0; i < N; i++) {
		result += to_string(arr[i]) + " ";
	}
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
	const int
		N = 4,
		PROCS_N = 4;

	MPI_Init(&argc, &argv);

	MPI_Comm MPI_COMM_RING;
	int dims[1] = { N };
	int periods[1] = { 1 };
	int reorder = 0;

	MPI_Cart_create(MPI_COMM_WORLD, 1, &dims[0], &periods[0], reorder, &MPI_COMM_RING);

	int
		A[N][N],
		B[N][N],
		C[N][N];

	int
		rowA[N],
		rowB[N];

	int rowC[N] = { 0 };

	const int offset = 10;

	int ringRank;
	MPI_Comm_rank(MPI_COMM_RING, &ringRank);

	if (ringRank == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = (N * i) + j + offset;
			}
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				B[i][j] = (N * i) + j + N * N + offset;
			}
		}

		cout << format("Матрица A: \n{}", matrixToStr(A)) << endl;
		cout << format("Матрица B: \n{}", matrixToStr(B)) << endl;
	}

	MPI_Scatter(&A[0], N, MPI_INT, &rowA[0], N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&B[0], N, MPI_INT, &rowB[0], N, MPI_INT, 0, MPI_COMM_WORLD);

	cout << format("Строка матрицы A, полученная процессом {}: \n{}", ringRank, arrToStr(rowA)) << endl;
	cout << format("Строка матрицы B, полученная процессом {}: \n{}", ringRank, arrToStr(rowB)) << endl;

	int left = (ringRank - 1 + PROCS_N) % PROCS_N;
	int right = (ringRank + 1) % PROCS_N;

	cout << format("Процесс-отправитель для процесса {}: {}", ringRank, left) << endl;
	cout << format("Процесс-получатель для процесса {}: {}", ringRank, right) << endl;

	for (int iter = 0; iter < N; iter++) {
		int i = (ringRank - iter + N) % N;
		for (int j = 0; j < N; j++) {
			rowC[j] += rowA[i] * rowB[j];
		}
		MPI_Sendrecv_replace(rowB, N, MPI_INT, right, 0, left, 0, MPI_COMM_RING, MPI_STATUS_IGNORE);
	}

	int localC[N][N] = { {0} };

	for (int i = 0; i < N; i++) {
		MPI_Sendrecv_replace(localC, N * N, MPI_INT, right, 0, left, 0, MPI_COMM_RING, MPI_STATUS_IGNORE);
		cout << format("Получаемая процессом {} матрица: \n{}", ringRank, matrixToStr(localC)) << endl;
		for (int j = 0; j < N; j++) {
			localC[ringRank][j] = rowC[j];
		}
		cout << format("Отправляемая процессом {} матрица: \n{}", ringRank, matrixToStr(localC)) << endl;
	}

	if (ringRank == 0) {
		cout << format("Результирующая матрица C: \n{}", matrixToStr(localC)) << endl;
	}

	MPI_Comm_free(&MPI_COMM_RING);
	MPI_Finalize();
	return 0;
}


