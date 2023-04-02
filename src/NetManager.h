#ifndef NETMANAGER_H
#define NETMANAGER_H

#include "NeuralNet.h"

#include <iostream>
#include <vector>

using namespace std;

class NetManager
{
public:
	NetManager(vector<unsigned> netsTopology, unsigned nbrOfNets);
	void update();
	NeuralNet* getBestNet(unsigned generation);

private:
	vector<NeuralNet*> nets, bestNets;

	unsigned generation;
};

#endif // NETMANAGER_H