#include <iostream>
#include <vector>
#include <queue>
#include "mpi.h"
#include <string>
#include <format>
using namespace std;

namespace {
	const int
		RS_NUM = 3,
		CL_NUM = 5,
		CL_BASE_RANK = 0,
		RS_BASE_RANK = CL_BASE_RANK + CL_NUM;

	enum MessageType {
		P = 1,
		V = 2,
		ACK = 3
	};

	struct Message {
		int type;
		int logicalClock;
		int clientID;
		int resourceID;
	};

	struct QueueItem {
		int logicalClock;
		int clientID;

		bool operator<(const QueueItem& other) const {
			if (logicalClock == other.logicalClock) return clientID > other.clientID;
			return logicalClock > other.logicalClock;
		}
	};

	string queueToStr(priority_queue<QueueItem> queue, int resourceID) {
		string result = format("Очередь ресурса {}:\n", resourceID);
		if (queue.empty()) result += "Очередь пуста\n";
		else {
			while (!queue.empty()) {
				QueueItem item = queue.top();
				queue.pop();

				result += format("LogicalClock: {}, Client {}; ", item.logicalClock, item.clientID);
			}
			result += "\n";
		}
		return result;
	}

	string queuesToStr(const vector<priority_queue<QueueItem>>& queues) {
		string result = "Состояние очередей ресурсов:\n";
		for (int resourceID = 0; resourceID < queues.size(); resourceID++) {
			result += queueToStr(queues[resourceID], resourceID);
		}
		return result;
	}

	void printQueue(priority_queue<QueueItem> queue, int resourceID) {
		cout << queueToStr(queue, resourceID) << endl;
	}

	void printQueues(const vector<priority_queue<QueueItem>>& queues) {
		cout << queuesToStr(queues) << endl;
	}

	void filterOutputByResource(string output, int currentResourceID, int targetResourceID = -1) {
		if (targetResourceID == -1) {
			cout << output << endl;
			return;
		}
		if (currentResourceID == targetResourceID) cout << output << endl;
	}

	int logicalClock = 0;
	int baseLogicalClock;
	vector<int> semaphores(RS_NUM, 1);
	vector<priority_queue<QueueItem>> resourceQueues(RS_NUM);

	vector<int> ackCount(RS_NUM, 0);
	vector<int> confirmationsP(RS_NUM, 0);
	vector<int> confirmationsV(RS_NUM, 0);

	void incrementLogicalClock() {
		logicalClock++;
	}

	void updateLogicalClock(int otherLogicalClock) {
		logicalClock = max(logicalClock, otherLogicalClock);
		incrementLogicalClock();
	}

	void sendBroadcast(Message msg, int srcClientID) {
		filterOutputByResource(
			format(
				"[Клиент {}]: [ЛЧ {}] до формирования широковещательного сообщения",
				logicalClock,
				srcClientID
			),
			msg.resourceID
		);
		incrementLogicalClock();
		filterOutputByResource(
			format(
				"[Клиент {}]: [ЛЧ {}] после формирования широковещательного сообщения",
				logicalClock,
				srcClientID
			),
			msg.resourceID
		);

		msg.logicalClock = logicalClock;
		baseLogicalClock = logicalClock;

		filterOutputByResource(
			format(
				"[Клиент {}] отправляет широковещательное сообщение с [Операция {}] для [Ресурс {}]",
				srcClientID,
				msg.type == 1 ? "P" : "V",
				msg.resourceID
			),
			msg.resourceID
		);

		for (int i = 0; i < CL_NUM; i++) {
			if (i == CL_BASE_RANK + srcClientID) continue;
			MPI_Send(&msg, sizeof(msg), MPI_BYTE, CL_BASE_RANK + i, 0, MPI_COMM_WORLD);
		}
	}

	void sendAck(int srcClientID, int destClientID, int resourceID) {
		filterOutputByResource(
			format(
				"[Клиент {}]: [ЛЧ {}] до формирования одноадресного подтверждающего сообщения для [Клиент {}]",
				srcClientID,
				logicalClock,
				destClientID
			),
			resourceID
		);
		incrementLogicalClock();
		filterOutputByResource(
			format(
				"[Клиент {}]: [ЛЧ {}] после формирования одноадресного подтверждающего сообщения для [Клиент {}]",
				srcClientID,
				logicalClock,
				destClientID
			),
			resourceID
		);

		Message msg;
		msg.type = ACK;
		msg.logicalClock = logicalClock;
		msg.clientID = srcClientID;
		msg.resourceID = resourceID;

		filterOutputByResource(
			format(
				"[Клиент {}] отправляет одноадресное подтверждающее сообщение [Клиент {}]",
				srcClientID,
				destClientID
			),
			msg.resourceID
		);

		MPI_Send(&msg, sizeof(msg), MPI_BYTE, CL_BASE_RANK + destClientID, 0, MPI_COMM_WORLD);
	}

	void recvMessage(int srcClientID, int destClientID) {
		Message msg;
		MPI_Recv(&msg, sizeof(msg), MPI_BYTE, CL_BASE_RANK + srcClientID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (msg.type == P || msg.type == V) {
			updateLogicalClock(msg.logicalClock);
			filterOutputByResource(
				format(
					"[Клиент {}]: Состояние очереди [Ресурс {}] до получения широковещательного сообщения c [Операция {}]:\n{}",
					destClientID,
					msg.resourceID,
					msg.type == P ? "P" : "V",
					queueToStr(resourceQueues[msg.resourceID], msg.resourceID)
				),
				msg.resourceID
			);

			if (msg.type == P) resourceQueues[msg.resourceID].push({ msg.logicalClock, msg.clientID });
			else {
				resourceQueues[msg.resourceID].pop();
				semaphores[msg.resourceID] = 1;
			}

			filterOutputByResource(
				format(
					"[Клиент {}]: Состояние очереди [Ресурс {}] после получения широковещательного сообщения c [Операция {}]:\n{}",
					destClientID,
					msg.resourceID,
					msg.type == P ? "P" : "V",
					queueToStr(resourceQueues[msg.resourceID], msg.resourceID)
				),
				msg.resourceID
			);
			sendAck(destClientID, srcClientID, msg.resourceID);
		}

		if (msg.type == ACK) {
			ackCount[msg.resourceID]++;
		}
	}

	void requestResource(int clientID, int resourceID) {
		Message msg;

		msg.type = P;
		msg.clientID = clientID;
		msg.resourceID = resourceID;

		filterOutputByResource(
			format(
				"[Клиент {}]: Состояние очереди [Ресурс {}] до получения подтверждения [Операция P]:\n{}",
				clientID,
				msg.resourceID,
				queueToStr(resourceQueues[msg.resourceID], msg.resourceID)
			),
			msg.resourceID
		);
		sendBroadcast(msg, clientID);
	}

	void captureResource(int clientID, int resourceID) {
		filterOutputByResource(
			format(
				"[Клиент {}] получил доступ к ресурсу [Ресурс {}]",
				clientID,
				resourceID
			),
			resourceID
		);
		semaphores[resourceID] = 0;
	}

	void releaseResource(int clientID, int resourceID) {
		Message msg;
		msg.type = V;
		msg.clientID = clientID;
		msg.resourceID = resourceID;

		filterOutputByResource(
			format(
				"[Клиент {}]: Состояние очереди [Ресурс {}] до получения подтверждения [Операция V]:\n{}",
				clientID,
				msg.resourceID,
				queueToStr(resourceQueues[msg.resourceID], msg.resourceID)
			),
			msg.resourceID
		);
		sendBroadcast(msg, clientID);
	}

	void enqueueSelf(int clientID, int resourceID) {
		filterOutputByResource(
			format(
				"[Клиент {}] получил все подтверждающие сообщения для [Операция P] для [Ресурс {}]",
				clientID,
				resourceID
			),
			resourceID
		);
		resourceQueues[resourceID].push({ baseLogicalClock, clientID });
		filterOutputByResource(
			format(
				"[Клиент {}]: Состояние очереди [Ресурс {}] после получения подтверждения [Операция P]:\n{}",
				clientID,
				resourceID,
				queueToStr(resourceQueues[resourceID], resourceID)
			),
			resourceID
		);
		confirmationsP[resourceID] = 1;
	}

	void dequeueSelf(int clientID, int resourceID) {
		filterOutputByResource(
			format(
				"[Клиент {}] получил все подтверждающие сообщения для [Операция V] для [Ресурс {}]",
				clientID,
				resourceID
			),
			resourceID
		);
		resourceQueues[resourceID].pop();
		filterOutputByResource(
			format(
				"[Клиент {}]: Состояние очереди [Ресурс {}] после получения подтверждения [Операция V]:\n{}",
				clientID,
				resourceID,
				queueToStr(resourceQueues[resourceID], resourceID)
			),
			resourceID
		);
		confirmationsV[resourceID] = 1;
		semaphores[resourceID] = 1;
	}

}

int alt(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int clientID = rank - CL_BASE_RANK;
	MPI_Status status;

	int requestsNum = rand() % 3 + 1;
	for (int i = 0; i < requestsNum; i++) {
		int resourceID = i;
		requestResource(clientID, resourceID);
		while (true) {
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
			if (flag) recvMessage(status.MPI_SOURCE - CL_BASE_RANK, clientID);

			if (ackCount[resourceID] == CL_NUM - 1) {
				if (semaphores[resourceID] == 1) enqueueSelf(clientID, resourceID);
				if (semaphores[resourceID] == 0) dequeueSelf(clientID, resourceID);
				ackCount[resourceID] = 0;
			}

			if (confirmationsP[resourceID]) {
				QueueItem top = resourceQueues[resourceID].top();
				if (top.clientID == clientID) {
					captureResource(clientID, resourceID);
					confirmationsP[resourceID] = 0;
					releaseResource(clientID, resourceID);
				}
				else {
					semaphores[resourceID] = 0;
				}
			}

			if (confirmationsV[resourceID]) {
				confirmationsV[resourceID] = 0;
				break;
			}
		}
	}

	MPI_Finalize();
	return 0;
}