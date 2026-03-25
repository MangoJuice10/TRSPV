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
	return LINE + str + LINE;
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
void halve(vector<T>& destArr, vector<T>& srcArr, bool firstHalf = true) {
	size_t half = destArr.size();

	if (firstHalf) copy(srcArr.begin(), srcArr.begin() + half, destArr.begin());
	else copy(srcArr.end() - half, srcArr.end(), destArr.begin());
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank,
		size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = static_cast<int>(log2(size));

	const int DATA_SIZE = 16;
	vector<int> data(DATA_SIZE);
	if (rank == 0) {
		srand(time(0));
		for (size_t i = 0; i < DATA_SIZE; i++) {
			data[i] = rand() % 100;
		}

		cout << addLines(format("Исходный массив: {}\n", arrToStr(data))) << endl;
	}

	const int BLOCK_SIZE = DATA_SIZE / size;
	vector<int> block(BLOCK_SIZE);
	vector<int> recvBlock(BLOCK_SIZE);
	vector<int> result(BLOCK_SIZE * 2);

	MPI_Scatter(data.data(), BLOCK_SIZE, MPI_INT, block.data(), BLOCK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	shellSort(block);

	if (rank == 0) {
		cout << addLines("Сортировка Шелла\n") << endl;
	}

	for (size_t i = 0; i < N; i++) {
		int partnerRank = rank ^ (1 << (N - i - 1));
		MPI_Sendrecv(block.data(), BLOCK_SIZE, MPI_INT, partnerRank, 0, recvBlock.data(), BLOCK_SIZE, MPI_INT, partnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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

		halve(block, result, rank < partnerRank);

		cout << format(
			"[Процесс{}]: Модифицированный локальный блок: {}\n\n",
			rank,
			arrToStr(block)
		);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		cout << addLines("Четная-Нечетная перестановка\n") << endl;
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

			MPI_Sendrecv(block.data(), BLOCK_SIZE, MPI_INT, partnerRank, 0, recvBlock.data(), BLOCK_SIZE, MPI_INT, partnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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

			halve(block, result, rank < partnerRank);

			cout << format(
				"[Процесс{}]: Модифицированный локальный блок: {}\n\n",
				rank,
				arrToStr(block)
			);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gather(block.data(), BLOCK_SIZE, MPI_INT, data.data(), BLOCK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		cout << addLines(format("Отсортированный массив: {}\n", arrToStr(data))) << endl;
	}

	MPI_Finalize();
	return 0;
}