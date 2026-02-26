#include <iostream>
#include <format>
#include "mpi.h"
using namespace std;

int main(int argc, char* argv[]) {
	int rank;
	int size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int num = 0;
	if (rank == 0) num = 2;
	if (rank == 1) num = 7;
	if (rank == 2) num = 5;
	if (rank == 3) num = -72;
	if (rank == 4) num = 100;

	int minNum = num;
	int maxNum = num;

	MPI_Request requests[4];
	int requestIdx = 0;
	for (int procIdx = 0; procIdx < size; procIdx++) {
		if (procIdx == rank) continue;
		MPI_Isend(&num, 1, MPI_INT, procIdx, 0, MPI_COMM_WORLD, &requests[requestIdx]);
		requestIdx++;
	}
	MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

	bool procStatuses[] = { true, true, true, true, true };
	procStatuses[rank] = false;

	int procIdx = 0;
	while (true) {
		if (!(procStatuses[0] || procStatuses[1] || procStatuses[2] || procStatuses[3] || procStatuses[4])) break;
		if (procStatuses[procIdx]) {
			int flag;
			MPI_Iprobe(procIdx, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
			if (flag) {
				int receivedNum;
				MPI_Recv(&receivedNum, 1, MPI_INT, procIdx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (minNum > receivedNum) minNum = receivedNum;
				if (maxNum < receivedNum) maxNum = receivedNum;
				procStatuses[procIdx] = false;
			}
		}
		procIdx = (procIdx + 1) % size;
	}

	cout << format("Локальный минимум процесса {}: {}", rank, minNum) << endl;
	cout << format("Локальный максимум процесса {}: {}", rank, maxNum) << endl;
	MPI_Finalize();

	return 0;
}