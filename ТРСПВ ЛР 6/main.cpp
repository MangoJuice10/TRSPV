#include <iostream>
#include "mpi.h"
#include <math.h>
#include <vector>
#include <string>
#include <format>
using namespace std;

template <typename T>
string arrToStr(vector<T> arr) {
	string result = "";
	for (T el : arr) {
		result += to_string(el) + " ";
	}
	return result;
}

string addLines(string str) {
	const string LINE = "------------------------------------------------------------------------------------------------------------------------------------------------\n";
	return LINE + str + "\n" + LINE;
}

template <typename T>
void shellSort(vector<T>& arr) {
	int n = arr.size();

	for (size_t gap = n / 2; gap > 0; gap /= 2) {
		for (size_t i = gap; i < n; i++) {
			size_t j = i;
			T temp = arr[j];
			while (j >= gap && arr[j - gap] > temp) {
				arr[j] = arr[j - gap];
				j -= gap;
			}
			arr[j] = temp;
		}
	}
}

int findPairProcess(int rank, int shift) {
	return rank ^ (1 << shift);
}

template <typename T>
void merge(vector<T>& result, vector<T>& arr1, vector<T>& arr2) {
	if (arr1.back() <= arr2.front()) {
		copy(arr1.begin(), arr1.end(), result.begin());
		copy(arr2.begin(), arr2.end(), result.begin() + arr1.size());
		return;
	}

	if (arr2.back() <= arr1.front()) {
		copy(arr2.begin(), arr2.end(), result.begin());
		copy(arr1.begin(), arr1.end(), result.begin() + arr2.size());
		return;
	}

	size_t
		i = 0,
		j = 0,
		k = 0;

	while (i < arr1.size() && j < arr2.size()) {
		if (arr1[i] <= arr2[j]) {
			result[k] = arr1[i];
			i++;
		}
		else {
			result[k] = arr2[j];
			j++;
		}
		k++;
	}

	while (i < arr1.size()) {
		result[k++] = arr1[i++];
	}

	while (j < arr2.size()) {
		result[k++] = arr2[j++];
	}
}

template<typename T>
void modify(vector<T>& destArr, vector<T>& srcArr, int offset, int size) {
	copy(srcArr.begin() + offset, srcArr.begin() + offset + size, destArr.begin());
}

template<typename T>
bool verifySortedArr(vector<T>& sortedArr, vector<T> srcArr) {
	shellSort(srcArr);
	return sortedArr == srcArr;
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank,
		size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	srand(time(0));

	int N = static_cast<int>(log2(size));

	const int BLOCK_SIZE = (rand() % 4 + 1);
	const int IRREGULARITY = (rand() % BLOCK_SIZE);
	const int IRREGULAR_BLOCK_SIZE = IRREGULARITY ? BLOCK_SIZE - IRREGULARITY : BLOCK_SIZE;

	const int DATA_SIZE = size * BLOCK_SIZE - IRREGULARITY;
	const int LOCAL_BLOCK_SIZE = rank == size - 1 ? IRREGULAR_BLOCK_SIZE : BLOCK_SIZE;

	vector<int> data(DATA_SIZE);
	if (rank == 0) {
		for (size_t i = 0; i < DATA_SIZE; i++) {
			data[i] = rand() % 100;
		}
		cout << addLines(format("Исходный массив: {}\nКоличество элементов в массиве: {}\nПоследний процесс имеет {} элементов, в то время как остальные имеют {} элементов", arrToStr(data), DATA_SIZE, IRREGULAR_BLOCK_SIZE, BLOCK_SIZE)) << endl;
	}

	vector<int> initialData = data;

	vector<int> block(LOCAL_BLOCK_SIZE);

	vector<int> sendCounts(size, BLOCK_SIZE);
	sendCounts[size - 1] = IRREGULAR_BLOCK_SIZE;

	vector<int> displs(size, 0);
	for (size_t i = 1; i < size; i++) {
		displs[i] = displs[i - 1] + sendCounts[i - 1];
	}

	const int LAST_PROCESS_RANK = size - 1;

	MPI_Scatterv(data.data(), sendCounts.data(), displs.data(), MPI_INT, block.data(), sendCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

	cout << format("[Процесс {}]: получен локальный блок: {}\n", rank, arrToStr(block)) << endl;

	shellSort(block);

	// Parallel Shell sort only applies to a hypercube
	if (size == static_cast<int>(pow(2, N))) {
		if (rank == 0) {
			cout << addLines("Сортировка Шелла") << endl;
		}

		for (size_t i = 0; i < N; i++) {
			int partnerRank = findPairProcess(rank, N - i - 1);

			const int SEND_BLOCK_SIZE =
				rank == LAST_PROCESS_RANK
				? IRREGULAR_BLOCK_SIZE
				: BLOCK_SIZE;

			const int RECV_BLOCK_SIZE =
				partnerRank == LAST_PROCESS_RANK
				? IRREGULAR_BLOCK_SIZE
				: BLOCK_SIZE;

			vector<int> recvBlock(RECV_BLOCK_SIZE);
			vector<int> result(SEND_BLOCK_SIZE + RECV_BLOCK_SIZE);

			MPI_Sendrecv(block.data(), SEND_BLOCK_SIZE, MPI_INT, partnerRank, 0, recvBlock.data(), RECV_BLOCK_SIZE, MPI_INT, partnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			merge(result, block, recvBlock);

			cout << format(
				"[Процесс {}] Локальный блок: {}\n[Процесс {}]: Получен блок от процесса {}: {}\n[Процесс {}]: Результирующий блок: {}\n\n",
				rank,
				arrToStr(block),
				rank,
				partnerRank,
				arrToStr(recvBlock),
				rank,
				arrToStr(result)
			);

			// Offset is always equal to the regular block size,
			// as the last process will always take the second part
			// of the resulting block.
			int offset = rank < partnerRank ? 0 : BLOCK_SIZE;
			int size = LOCAL_BLOCK_SIZE;
			modify(block, result, offset, size);

			cout << format(
				"[Процесс {}]: Модифицированный локальный блок: {}\n\n",
				rank,
				arrToStr(block)
			);
			MPI_Barrier(MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank == 0) {
		cout << addLines("Четная-Нечетная перестановка") << endl;
	}

	for (size_t i = 0; i < size; i++) {
		bool isIterationEven = i % 2 == 0;
		bool isProcessRankEven = rank % 2 == 0;
		bool isSizeEven = size % 2 == 0;

		bool participates = true;
		if (rank == 0 && !isIterationEven) participates = false;
		if (rank == size - 1) {
			if (isIterationEven && !isSizeEven) participates = false;
			if (!isIterationEven && isSizeEven) participates = false;
		}

		if (participates) {
			int partnerRank = rank + (isIterationEven
				? (isProcessRankEven ? 1 : -1)
				: (isProcessRankEven ? -1 : 1));

			const int SEND_BLOCK_SIZE =
				rank == LAST_PROCESS_RANK
				? IRREGULAR_BLOCK_SIZE
				: BLOCK_SIZE;

			const int RECV_BLOCK_SIZE =
				partnerRank == LAST_PROCESS_RANK
				? IRREGULAR_BLOCK_SIZE
				: BLOCK_SIZE;

			vector<int> recvBlock(RECV_BLOCK_SIZE);
			vector<int> result(SEND_BLOCK_SIZE + RECV_BLOCK_SIZE);

			MPI_Sendrecv(block.data(), SEND_BLOCK_SIZE, MPI_INT, partnerRank, 0, recvBlock.data(), RECV_BLOCK_SIZE, MPI_INT, partnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			merge(result, block, recvBlock);
			cout << format(
				"[Процесс {}] Локальный блок: {}\n[Процесс {}]: Получен блок от процесса {}: {}\n[Процесс {}]: Результирующий блок: {}\n\n",
				rank,
				arrToStr(block),
				rank,
				partnerRank,
				arrToStr(recvBlock),
				rank,
				arrToStr(result)
			);

			int offset = rank < partnerRank ? 0 : BLOCK_SIZE;
			int size = LOCAL_BLOCK_SIZE;
			modify(block, result, offset, size);

			cout << format(
				"[Процесс{}]: Модифицированный локальный блок: {}\n\n",
				rank,
				arrToStr(block)
			);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gatherv(block.data(), LOCAL_BLOCK_SIZE, MPI_INT, data.data(), sendCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		if (verifySortedArr(data, initialData)) {
			cout << addLines(format("Отсортированный массив: {}\nКоличество элементов в массиве: {}\nПоследний процесс имеет {} элементов, в то время как остальные имеют {} элементов", arrToStr(data), DATA_SIZE, IRREGULAR_BLOCK_SIZE, BLOCK_SIZE)) << endl;
		}
		else {
			cout << addLines(format("Массив был отсортирован неправильно: {}", arrToStr(data))) << endl;
		}
	}

	MPI_Finalize();
	return 0;
}