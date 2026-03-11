#include <iostream>
#include "mpi.h"
#include <queue>
#include <vector>
#include <string>
#include <format>
using namespace std;

namespace {
	const int DATA_SIZE = 2;

	const int CL_NUM = 5;
	const int RS_NUM = 3;
	const int SR_BASE_RANK = 0;
	const int CL_BASE_RANK = 1;
	const int RS_BASE_RANK = CL_BASE_RANK + CL_NUM;

	const int REQUEST = 1;
	const int RELEASE = 2;
	const int RESPONSE = 3;

	struct Message {
		int type;
		int clientID;
		int resourceID;
	};

	template <typename T, size_t N>
	string arrToStr(T(&arr)[N]) {
		string result = "";
		for (size_t i = 0; i < N; i++) {
			result += to_string(arr[i]) + " ";
		}
		return result;
	}

	string queueToStr(queue<Message> queue, int resourceID) {
		string result = format("Состояние очереди ресурса {}:\n", resourceID);
		if (queue.empty()) result += "Очередь пуста\n";
		else {
			while (!queue.empty()) {
				Message req = queue.front();
				queue.pop();
				result += format("Client: {}; ", req.clientID);
			}
			result += "\n";
		}
		return result;
	}

	void printQueue(queue<Message> queue, int resourceID) {
		cout << queueToStr(queue, resourceID) << endl;
	}

	string queuesToStr(const vector<queue<Message>>& queues) {
		string result = "Состояние очередей ресурсов:\n";
		for (int resourceID = 0; resourceID < queues.size(); resourceID++) {
			result += format("Очередь ресурса {}:\n", resourceID);

			queue<Message> queue = queues[resourceID];
			result += queueToStr(queue, resourceID);
		}
		return result;
	}

	void printQueues(const vector<queue<Message>>& queues) {
		cout << queuesToStr(queues) << endl;
	}

	void filterOutputByResource(string output, int currentResourceID, int targetResourceID = 1) {
		if (targetResourceID == -1) {
			cout << output << endl;
			return;
		}

		if (currentResourceID == targetResourceID) cout << output << endl;
	}

	void client1(int rank) {
		int clientID = rank - CL_BASE_RANK;
		int sum = 0;
		for (size_t resourceID = 0; resourceID < RS_NUM; resourceID++) {
			Message req;
			req.type = REQUEST;
			req.clientID = clientID;
			req.resourceID = resourceID;
			MPI_Send(&req, sizeof(req), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD);
			filterOutputByResource(format("Клиентом {} отправлен запрос на использование ресурса {}", clientID, resourceID), resourceID);
			Message res;
			MPI_Recv(&res, sizeof(res), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			filterOutputByResource(format("Клиентом {} получено подтверждение на использование ресурса {}", clientID, resourceID), resourceID);

			int data[DATA_SIZE];
			data[0] = clientID;
			data[1] = resourceID;
			MPI_Send(&data, DATA_SIZE, MPI_INT, RS_BASE_RANK + resourceID, 0, MPI_COMM_WORLD);
			filterOutputByResource(format("Клиентом {} отправлено сообщение ресурсу {} с данными {}", clientID, resourceID, arrToStr(data)), resourceID);
			int result;
			MPI_Recv(&result, 1, MPI_INT, RS_BASE_RANK + resourceID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			filterOutputByResource(format("Клиентом {} получено сообщение от ресурса {} с данными: {}", clientID, resourceID, result), resourceID);
			sum += result;

			Message release;
			release.type = RELEASE;
			release.resourceID = resourceID;
			MPI_Send(&release, sizeof(release), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD);
			filterOutputByResource(format("Клиентом {} был освобожден ресурс {}", clientID, resourceID), resourceID);
		}
		cout << "Итоговая сумма, вычисленная клиентом " << clientID << ": " << sum << endl;
	}

	void resource1(int rank) {
		int resourceID = rank - RS_BASE_RANK;
		while (true) {
			MPI_Status status;
			int data[DATA_SIZE];
			MPI_Recv(&data, DATA_SIZE, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			filterOutputByResource(format("Ресурсом {} были получены данные {}", resourceID, arrToStr(data)), resourceID);
			int clientRank = status.MPI_SOURCE;
			int result = 0;
			for (size_t i = 0; i < DATA_SIZE; i++) {
				result += data[i];
			}

			MPI_Send(&result, 1, MPI_INT, clientRank, 0, MPI_COMM_WORLD);
			filterOutputByResource(format("Ресурсом {} были отправлены данные {}", resourceID, result), resourceID);
		}
	}

	void server1() {
		vector<queue<Message>> queues(RS_NUM);

		while (true) {
			Message msg;
			MPI_Recv(&msg, sizeof(msg), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			int resourceID = msg.resourceID;

			if (msg.type == REQUEST) {
				bool isEmpty = queues[resourceID].empty();
				queues[resourceID].push(msg);
				filterOutputByResource(format("Сервер получил запрос клиента {} на ресурс {}\n{}",
					msg.clientID,
					resourceID,
					queueToStr(queues[resourceID], resourceID)
				), resourceID);

				if (isEmpty) {
					Message res;
					res.type = RESPONSE;
					MPI_Send(&res, sizeof(res), MPI_BYTE, CL_BASE_RANK + msg.clientID, 0, MPI_COMM_WORLD);
				}
			}
			else if (msg.type == RELEASE) {
				queues[resourceID].pop();
				bool isEmpty = queues[resourceID].empty();

				if (!isEmpty) {
					Message req = queues[resourceID].front();

					Message res;
					res.type = RESPONSE;
					filterOutputByResource(format("Сервер отправляет подтверждение на запрос к ресурсу {} для клиента {}\n{}",
						resourceID,
						req.clientID,
						queueToStr(queues[resourceID], resourceID)
					), resourceID);
					MPI_Send(&res, sizeof(res), MPI_BYTE, CL_BASE_RANK + req.clientID, 0, MPI_COMM_WORLD);
				}
			}
		}
	}
}


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) server1();
	if (rank >= 1 && rank < CL_BASE_RANK + CL_NUM) client1(rank);
	if (rank >= RS_BASE_RANK && rank < RS_BASE_RANK + RS_NUM) resource1(rank);
	MPI_Finalize();
	return 0;
}