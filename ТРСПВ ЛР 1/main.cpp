#include <iostream> 
#include <string> 
#include <format>
#include "mpi.h"
using namespace std;

template <size_t R, size_t C>
string matrixToStr(int(&matrix)[R][C]) {
	string result = "";
	for (size_t i = 0; i < R; i++) {
		for (size_t j = 0; j < C; j++) {
			result += to_string(matrix[i][j]) + " ";
		}
		result += "\n";
	}
	return result;
}

template <size_t R, size_t C>
void showMatrix(int(&matrix)[R][C]) {
	cout << matrixToStr(matrix) << endl;
}

template <size_t R, size_t C>
void findMinor(
	int(&matrix)[R][C],
	int(&minor)[R - 1][C - 1],
	size_t removedRow,
	size_t removedCol
) {
	int minorRow = 0;
	for (size_t matrixRow = 0; matrixRow < R; matrixRow++) {
		int minorCol = 0;
		if (matrixRow == removedRow) continue;
		for (size_t matrixCol = 0; matrixCol < C; matrixCol++) {
			if (matrixCol == removedCol) continue;
			minor[minorRow][minorCol++] = matrix[matrixRow][matrixCol];
		}
		minorRow++;
	}
}

int findDeterminant3x3(int(&matrix)[3][3]) {
	int result =
		matrix[0][0] * matrix[1][1] * matrix[2][2] +
		matrix[0][2] * matrix[1][0] * matrix[2][1] +
		matrix[0][1] * matrix[1][2] * matrix[2][0] -
		matrix[0][2] * matrix[1][1] * matrix[2][0] -
		matrix[0][1] * matrix[1][0] * matrix[2][2] -
		matrix[0][0] * matrix[1][2] * matrix[2][1];
	return result;
}

template <size_t R, size_t C>
int findAlgebraicComplement(int(&matrix)[R][C], size_t removedRow, size_t removedCol) {
	int multiplier = (((removedRow + removedCol) % 2) == 0 ? 1 : -1) * matrix[removedRow][removedCol];
	int minor[R - 1][C - 1];
	findMinor(matrix, minor, removedRow, removedCol);
	int determinant = findDeterminant3x3(minor);
	return multiplier * determinant;
}

int main(int argc, char* argv[])
{
	const int
		ROWS_NUM = 4,
		COLS_NUM = 4;
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double
		startTime,
		endTime;

	startTime = MPI_Wtime();

	if (rank == 0) {
		// Матрица 4x4, для которой необходимо будет найти определитель
		/*int matrix[ROWS_NUM][COLS_NUM] = {
			{1, 2, 3, 4},
			{5, 6, 7, 8},
			{9, 1, 2, 3},
			{4, 5, 6, 7}
		};*/

		int matrix[ROWS_NUM][COLS_NUM] = {
			{14, 0, 15, 4},
			{5, 6, 37, 8},
			{2, 9, 2, 3},
			{4, 35, 6, 7}
		};

		cout << "Исходная матрица:" << endl;
		showMatrix(matrix);

		MPI_Request requests[4];

		for (int i = 1; i < size; i++) {
			MPI_Isend(&matrix[0][0], ROWS_NUM * COLS_NUM, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
		}
		MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);

		int determinant = 0;

		bool procStatuses[4] = { true, true, true, true };
		int procIdx = 1;
		while (true) {
			if (!(procStatuses[0] || procStatuses[1] || procStatuses[2] || procStatuses[3])) break;
			if (procStatuses[procIdx - 1]) {
				int flag;
				MPI_Iprobe(procIdx, 0, MPI_COMM_WORLD, &flag, MPI_STATUSES_IGNORE);
				if (flag) {
					int algebraicComplement;
					MPI_Recv(&algebraicComplement, 1, MPI_INT, procIdx, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

					cout << format("Алгебраическое дополнение для процесса {}: {}", procIdx, algebraicComplement) << endl;

					determinant += algebraicComplement;
					procStatuses[procIdx - 1] = false;
				}
			}
			procIdx = procIdx % 4 + 1;
		}
		cout << "Определитель матрицы 4x4: " << determinant << endl;
		endTime = MPI_Wtime();
		cout << format("Прошедшее с момента запуска программы время: {} секунд", endTime - startTime) << endl;
	}

	int matrix[ROWS_NUM][COLS_NUM];
	int row,
		col;

	if (rank == 1) {
		row = 0;
		col = 0;
		MPI_Status messageStatus;
		MPI_Recv(&matrix[0][0], ROWS_NUM * COLS_NUM, MPI_INT, 0, 0, MPI_COMM_WORLD, &messageStatus);
		int result = findAlgebraicComplement(matrix, row, col);
		MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 2) {
		row = 0,
		col = 1;
		MPI_Status messageStatus;
		MPI_Recv(&matrix[0][0], ROWS_NUM * COLS_NUM, MPI_INT, 0, 0, MPI_COMM_WORLD, &messageStatus);
		int result = findAlgebraicComplement(matrix, row, col);
		MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 3) {
		row = 0;
		col = 2;
		MPI_Status messageStatus;
		MPI_Recv(&matrix[0][0], ROWS_NUM * COLS_NUM, MPI_INT, 0, 0, MPI_COMM_WORLD, &messageStatus);
		int result = findAlgebraicComplement(matrix, row, col);
		MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 4) {
		row = 0;
		col = 3;
		MPI_Status messageStatus;
		MPI_Recv(&matrix[0][0], ROWS_NUM * COLS_NUM, MPI_INT, 0, 0, MPI_COMM_WORLD, &messageStatus);
		int result = findAlgebraicComplement(matrix, row, col);
		MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}