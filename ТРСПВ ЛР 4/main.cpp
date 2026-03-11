#include <iostream>
#include "mpi.h"
#include <vector>
#include <queue>
#include <format>
#include <string>
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
	const int TASK = 4;
	const int RESULT = 5;

	struct Message {
		int type;
		int clientID;
		int resourceID;
		int data[DATA_SIZE];
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
				result += format("Client: {}, Data: {}; ", req.clientID, arrToStr(req.data));
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

	void client2(int rank) {
		int clientID = rank - CL_BASE_RANK;
		int sum = 0;

		const int REQ_RS_NUM = RS_NUM;

		MPI_Request requests[REQ_RS_NUM];
		for (int resourceID = 0; resourceID < REQ_RS_NUM; resourceID++) {
			Message req;
			req.type = REQUEST;
			req.clientID = clientID;
			req.resourceID = resourceID;
			req.data[0] = clientID;
			req.data[1] = resourceID;
			MPI_Isend(&req, sizeof(req), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD, &requests[resourceID]);
			filterOutputByResource(format("Клиентом {} отправлено сообщение ресурсу {} с данными {}", clientID, req.resourceID, arrToStr(req.data)), resourceID);
		}
		MPI_Waitall(REQ_RS_NUM, requests, MPI_STATUSES_IGNORE);

		int received = 0;
		while (received < REQ_RS_NUM) {
			int flag;
			MPI_Iprobe(SR_BASE_RANK, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
			if (flag) {
				Message res;
				MPI_Recv(&res, sizeof(res), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (res.type == RESPONSE) {
					filterOutputByResource(format("Клиентом {} получено сообщение от ресурса {} с данными: {}", clientID, res.resourceID, arrToStr(res.data)), res.resourceID);
					received++;
					sum += res.data[0];
				}
			}
		}

		cout << "Итоговая сумма, вычисленная клиентом " << clientID << " : " << sum << endl;
	}

	void resource2(int rank) {
		int resourceID = rank - RS_BASE_RANK;
		while (true) {
			Message req;
			MPI_Recv(&req, sizeof(req), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (req.type == TASK) {
				filterOutputByResource(format("Ресурсом {} получено сообщение с данными {}", resourceID, arrToStr(req.data)), resourceID);
				int result = 0;
				for (size_t i = 0; i < DATA_SIZE; i++) {
					result += req.data[i];
				}

				Message res;
				res.type = RESULT;
				res.resourceID = resourceID;
				res.data[0] = result;
				for (size_t i = 1; i < DATA_SIZE; i++) res.data[i] = 0;
				MPI_Send(&res, sizeof(res), MPI_BYTE, SR_BASE_RANK, 0, MPI_COMM_WORLD);
				filterOutputByResource(format("Ресурсом {} отправлено сообщение с данными {}", resourceID, arrToStr(res.data)), resourceID);
			}
		}
	}

	void server2() {
		vector<queue<Message>> queues(RS_NUM);
		printQueues(queues);

		while (true) {
			Message msg;
			MPI_Recv(&msg, sizeof(msg), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (msg.type == REQUEST) {
				int resourceID = msg.resourceID;
				bool isEmpty = queues[resourceID].empty();
				queues[resourceID].push(msg);
				filterOutputByResource(format("Сервер получил запрос клиента {} на ресурс {}\n{}",
					msg.clientID,
					resourceID,
					queueToStr(queues[resourceID], resourceID)
				), resourceID);

				if (isEmpty) {
					Message sentReq;
					sentReq.type = TASK;
					for (size_t i = 0; i < DATA_SIZE; i++) {
						sentReq.data[i] = msg.data[i];
					}
					MPI_Send(&sentReq, sizeof(sentReq), MPI_BYTE, RS_BASE_RANK + resourceID, 0, MPI_COMM_WORLD);
				}
			}
			else if (msg.type == RESULT) {
				int resourceID = msg.resourceID;
				Message recvReq = queues[resourceID].front();
				queues[resourceID].pop();
				filterOutputByResource(format("Сервер получил ответ ресурса {} для клиента {}\n{}",
					resourceID,
					recvReq.clientID,
					queueToStr(queues[resourceID], resourceID)
				), resourceID);

				Message sentRes;
				sentRes.type = RESPONSE;
				sentRes.resourceID = resourceID;
				sentRes.data[0] = msg.data[0];
				for (size_t i = 0; i < DATA_SIZE; i++) {
					sentRes.data[i] = msg.data[i];
				}
				MPI_Send(&sentRes, sizeof(sentRes), MPI_BYTE, CL_BASE_RANK + recvReq.clientID, 0, MPI_COMM_WORLD);

				bool isEmpty = queues[resourceID].empty();
				if (!isEmpty) {
					Message nextRecvReq = queues[resourceID].front();
					Message nextSentReq;
					nextSentReq.type = TASK;
					for (size_t i = 0; i < DATA_SIZE; i++) {
						nextSentReq.data[i] = nextRecvReq.data[i];
					}
					MPI_Send(&nextSentReq, sizeof(nextSentReq), MPI_BYTE, RS_BASE_RANK + resourceID, 0, MPI_COMM_WORLD);
				}
			}
		}
	}

}

int alt(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == SR_BASE_RANK) server2();

	if (rank >= CL_BASE_RANK && rank < CL_BASE_RANK + CL_NUM) client2(rank);

	if (rank >= RS_BASE_RANK && rank < RS_BASE_RANK + RS_NUM) resource2(rank);

	MPI_Finalize();
	return 0;
}