#include "NetManager.h"

NetManager::NetManager(vector<unsigned> netsTopology, unsigned nbrOfNets) : generation(0)
{
	for (unsigned netNum = 0; netNum < nbrOfNets; netNum++)
		this->nets.push_back(new NeuralNet(netsTopology));
}

void NetManager::update()
{
	bool fin = true;

	for (NeuralNet* net : this->nets)
		if (net->isAlive) {
			fin = false;
			break;
		}

	if (!fin)
		for (NeuralNet* net : this->nets)
			net->update(generation < 500 ? 100 : generation < 1000 ? 200 : 1200);
	else {
		std::sort(this->nets.begin(), this->nets.end(), [](NeuralNet* n1, NeuralNet* n2) {
			return n1->score > n2->score;
		});

		this->bestNets.push_back(this->nets[0]);

		vector<NeuralNet*> newNets;
		double mutationRates[4] = { 1.0, 3.0, 10.0, 15.0 };

		for (unsigned i = 0; i < 4; i++)
			for (unsigned j = 0; j < (double)this->nets.size() / 4.0; j++)
				newNets.push_back(new NeuralNet(*this->nets[j], mutationRates[i]));

		this->nets = newNets;

		cout << "G\202n\202ration max : " << this->generation << endl;

		this->generation++;
	}
}

NeuralNet* NetManager::getBestNet(unsigned generation)
{
	return (generation < this->generation) ? this->bestNets[generation] : NULL;
}