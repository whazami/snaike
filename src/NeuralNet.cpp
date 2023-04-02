#include "NeuralNet.h"

#include <random>
#include <ctime>

#include <functional>
#include <algorithm>

NeuralNet::NeuralNet(vector<unsigned> topology) : topology(topology), score(0), snakePoses({ sf::Vector2u(SIZE_COTE / 2, SIZE_COTE / 2) }), snakeDir(DROITE), pommePoses({ sf::Vector2u(rand() % SIZE_COTE, rand() % SIZE_COTE) }), isAlive(true), sizes({ 1 }), movesUntilNoAppleEaten(0)
{
	vector<unsigned> weightMatsDims;
	for (unsigned i = 0; i < topology.size() - 1; i++) {
		weightMatsDims.push_back(topology[i + 1]);
		weightMatsDims.push_back(topology[i]);
	}

	default_random_engine generator((int)time(NULL));
	normal_distribution<double> randLoiNormale(0.0, 1.0);

	for (unsigned matNum = 0; matNum < weightMatsDims.size() / 2; matNum++) {
		this->weightMats.push_back(vector<vector<double>>());
		for (int matLine = 0; matLine < weightMatsDims[2 * matNum]; matLine++) {
			this->weightMats[matNum].push_back(vector<double>());
			for (int matCol = 0; matCol < weightMatsDims[2 * matNum + 1]; matCol++) {
				this->weightMats[matNum][matLine].push_back(randLoiNormale(generator) / sqrt(weightMatsDims[2 * matNum + 1]));	// Afin d'obtenir un apprentissage plus rapide, plus il y a de neurones d'entr�e sur la matrice actuelle, plus on r�duit le nombre al�atoire g�n�r�
			}
		}
	}

	for (unsigned layerNum = 1; layerNum < topology.size(); layerNum++) {														// On rappelle que l'inputLayer n'a pas de bias
		this->biases.push_back(vector<double>());
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
			this->biases[layerNum - 1].push_back(0.0);
		}
	}
}

NeuralNet::NeuralNet(NeuralNet& net, double randomWeightsBiasesPercentage) : topology(net.topology), score(0), snakePoses({ sf::Vector2u(SIZE_COTE / 2, SIZE_COTE / 2) }), snakeDir(DROITE), pommePoses({ sf::Vector2u(rand() % SIZE_COTE, rand() % SIZE_COTE) }), isAlive(true), sizes({ 1 }), movesUntilNoAppleEaten(0)
{
	for (unsigned matNum = 0, layerNum = 1; matNum < net.weightMats.size(); matNum++, layerNum++) {
		this->weightMats.push_back(vector<vector<double>>());
		this->biases.push_back(vector<double>());
		for (unsigned matLine = 0, neuronNum = 0; matLine < net.weightMats[matNum].size(); matLine++, neuronNum++) {
			this->weightMats[matNum].push_back(vector<double>());

			if ((double)rand() / (double)RAND_MAX * 100.0 < randomWeightsBiasesPercentage)
				this->biases[layerNum - 1].push_back(((rand() % 2 == 0) ? 1.0 : -1.0) * (double)rand() / (double)RAND_MAX);
			else
				this->biases[layerNum - 1].push_back(net.biases[layerNum - 1][neuronNum]);

			for (unsigned matCol = 0; matCol < net.weightMats[matNum][matLine].size(); matCol++) {
				if ((double)rand() / (double)RAND_MAX * 100.0 < randomWeightsBiasesPercentage)
					this->weightMats[matNum][matLine].push_back(((rand() % 2 == 0) ? 1.0 : -1.0) * (double)rand() / (double)RAND_MAX);
				else
					this->weightMats[matNum][matLine].push_back(net.weightMats[matNum][matLine][matCol]);
			}
		}
	}
}

NeuralNet::NeuralNet(string fileName) : score(0), snakePoses({ sf::Vector2u(SIZE_COTE / 2, SIZE_COTE / 2) }), snakeDir(DROITE), pommePoses({ sf::Vector2u(rand() % SIZE_COTE, rand() % SIZE_COTE) }), isAlive(true), sizes({ 1 }), movesUntilNoAppleEaten(0)
{
	ifstream rFile(fileName);

	if (rFile) {
		string topologyStr;
		getline(rFile, topologyStr);

		if (topologyStr.find("Topology: ") == string::npos) {
			cout << "Erreur : Topologie inexistante dans le fichier " << fileName << "." << endl;
			return;
		}

		topologyStr.erase(0, 9);
		stringstream ss;
		ss << topologyStr;

		unsigned nb;
		while (ss >> nb) this->topology.push_back(nb);

		vector<unsigned> weightMatsDims;
		for (unsigned i = 0; i < this->topology.size() - 1; i++) {
			weightMatsDims.push_back(this->topology[i + 1]);
			weightMatsDims.push_back(this->topology[i]);
		}

		for (unsigned matNum = 0; matNum < weightMatsDims.size() / 2; matNum++) {
			this->weightMats.push_back(vector<vector<double>>());
			for (int matLine = 0; matLine < weightMatsDims[2 * matNum]; matLine++) {
				this->weightMats[matNum].push_back(vector<double>());
				for (int matCol = 0; matCol < weightMatsDims[2 * matNum + 1]; matCol++) {
					double weight;
					rFile >> weight;
					this->weightMats[matNum][matLine].push_back(weight);
				}
			}
		}

		for (unsigned layerNum = 1; layerNum < this->topology.size(); layerNum++) {												// On rappelle que l'inputLayer n'a pas de bias
			this->biases.push_back(vector<double>());
			for (unsigned neuronNum = 0; neuronNum < this->topology[layerNum]; neuronNum++) {
				double bias;
				rFile >> bias;
				this->biases[layerNum - 1].push_back(bias);
			}
		}

		rFile.close();
	}
	else
		cout << "Erreur : Impossible d'ouvrir le fichier '" << fileName << "' en lecture." << endl;
}

vector<double> NeuralNet::getResults(vector<double> inputs)
{
	// S�curit�
	if (inputs.size() != this->topology[0]) {
		cout << "Erreur : Le nombre d'entr\202es donn\202es au r\202seau ne correspond pas au nombre d'entr\202es de celui-ci." << endl;
		exit(-1);
	}

	// Feed forward
	vector<vector<double>> z, a;
	a.push_back(inputs);

	for (unsigned layerNum = 1; layerNum < this->topology.size(); layerNum++) {
		z.push_back(a.back() * this->weightMats[layerNum - 1] + this->biases[layerNum - 1]);
		
		a.push_back(vector<double>());
		for (unsigned neuronNum = 0; neuronNum < z.back().size(); neuronNum++)
			a.back().push_back(NeuralNet::activation(z.back()[neuronNum]));
	}

	return a.back();
}

double NeuralNet::activation(double x, bool derivative)
{
	return (!derivative) ? tanh(x) : 1.0 - pow(tanh(x), 2);
}

void NeuralNet::save(string fileName)
{
	ofstream wFile(fileName);

	if (wFile) {
		wFile << "Topology: ";
		for (unsigned layerInfo : this->topology)
			wFile << layerInfo << " ";
		wFile << endl << endl;

		vector<unsigned> weightMatsDims;
		for (unsigned i = 0; i < this->topology.size() - 1; i++) {
			weightMatsDims.push_back(this->topology[i + 1]);
			weightMatsDims.push_back(this->topology[i]);
		}

		for (unsigned matNum = 0; matNum < weightMatsDims.size() / 2; matNum++) {
			for (int matLine = 0; matLine < weightMatsDims[2 * matNum]; matLine++) {
				for (int matCol = 0; matCol < weightMatsDims[2 * matNum + 1]; matCol++)
					wFile << this->weightMats[matNum][matLine][matCol] << " ";
				wFile << endl;
			}
			wFile << endl;
		}

		for (unsigned layerNum = 1; layerNum < this->topology.size(); layerNum++) {
			for (unsigned neuronNum = 0; neuronNum < this->topology[layerNum]; neuronNum++)
				wFile << this->biases[layerNum - 1][neuronNum] << " ";
			wFile << endl;
		}

		wFile.close();
	}
	else
		cout << "Erreur : Impossible d'ouvrir le fichier '" << fileName << "' en \202criture." << endl;
}

void NeuralNet::update(unsigned int movesLimitUntilNoAppleEaten)
{
	if (this->isAlive)
	{
		vector<double> inputs;
		/* Entr�es 1, 2, 3, 4, 5 : pr�sence de pomme ou non respectivement en face, � droite, � gauche, en face � droite et en face � gauche (si pr�sence --> distance)
		 * Entr�es 6, 7, 8 : pr�sence d'obstacle (mur ou queue) ou non respectivement en face, � droite et � gauche (si pr�sence --> distance (fonctionnel) */
		sf::Vector2u vectDiff = this->pommePoses.back() - this->snakePoses.back();
		double minDistFace = 1000, minDistDroite = 1000, minDistGauche = 1000;
		switch (this->snakeDir) {
		case HAUT:
			// Pomme inputs
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y < this->snakePoses.back().y) inputs.push_back((this->snakePoses.back().y - this->pommePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x > this->snakePoses.back().x) inputs.push_back((this->pommePoses.back().x - this->snakePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x < this->snakePoses.back().x) inputs.push_back((this->snakePoses.back().x - this->pommePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (vectDiff.x == vectDiff.y && vectDiff.y < 0) {
				inputs.push_back((vectDiff.x > 0) ? vectDiff.x / 10.0 : -1.0);
				inputs.push_back((vectDiff.x < 0) ? vectDiff.x / 10.0 : -1.0);
			}
			else {
				inputs.push_back(-1.0);
				inputs.push_back(-1.0);
			}

			// Mur & queue inputs
			for (unsigned i = this->snakePoses.size() - 2; i > this->snakePoses.size() - 1 - this->sizes.back(); i--) {
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y < this->snakePoses.back().y) {
					double dist = (this->snakePoses.back().y - this->snakePoses[i].y) / 10.0;
					if (dist < minDistFace)
						minDistFace = dist;
				}
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x > this->snakePoses.back().x) {
					double dist = (this->snakePoses[i].x - this->snakePoses.back().x) / 10.0;
					if (dist < minDistDroite)
						minDistDroite = dist;
				}
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x < this->snakePoses.back().x) {
					double dist = (this->snakePoses.back().x - this->snakePoses[i].x) / 10.0;
					if (dist < minDistGauche)
						minDistGauche = dist;
				}
			}
			inputs.push_back((minDistFace != 1000) ? minDistFace : this->snakePoses.back().y / 10.0);
			inputs.push_back((minDistDroite != 1000) ? minDistDroite : ((SIZE_COTE - 1) - this->snakePoses.back().x) / 10.0);
			inputs.push_back((minDistGauche != 1000) ? minDistGauche : this->snakePoses.back().x / 10.0);
			break;

		case BAS:
			// Pomme inputs
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y > this->snakePoses.back().y) inputs.push_back((this->pommePoses.back().y - this->snakePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x < this->snakePoses.back().x) inputs.push_back((this->snakePoses.back().x - this->pommePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x > this->snakePoses.back().x) inputs.push_back((this->pommePoses.back().x - this->snakePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (vectDiff.x == vectDiff.y && vectDiff.y > 0) {
				inputs.push_back((vectDiff.x < 0) ? vectDiff.x / 10.0 : -1.0);
				inputs.push_back((vectDiff.x > 0) ? vectDiff.x / 10.0 : -1.0);
			}
			else {
				inputs.push_back(-1.0);
				inputs.push_back(-1.0);
			}

			// Mur & queue inputs
			for (unsigned i = this->snakePoses.size() - 2; i > this->snakePoses.size() - 1 - this->sizes.back(); i--) {
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y > this->snakePoses.back().y) {
					double dist = (this->snakePoses[i].y - this->snakePoses.back().y) / 10.0;
					if (dist < minDistFace)
						minDistFace = dist;
				}
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x < this->snakePoses.back().x) {
					double dist = (this->snakePoses.back().x - this->snakePoses[i].x) / 10.0;
					if (dist < minDistDroite)
						minDistDroite = dist;
				}
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x > this->snakePoses.back().x) {
					double dist = (this->snakePoses[i].x - this->snakePoses.back().x) / 10.0;
					if (dist < minDistGauche)
						minDistGauche = dist;
				}
			}
			inputs.push_back((minDistFace != 1000) ? minDistFace : ((SIZE_COTE - 1) - this->snakePoses.back().y) / 10.0);
			inputs.push_back((minDistDroite != 1000) ? minDistDroite : this->snakePoses.back().x / 10.0);
			inputs.push_back((minDistGauche != 1000) ? minDistGauche : ((SIZE_COTE - 1) - this->snakePoses.back().x) / 10.0);
			break;

		case GAUCHE:
			// Pomme inputs
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x < this->snakePoses.back().x) inputs.push_back((this->snakePoses.back().x - this->pommePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y < this->snakePoses.back().y) inputs.push_back((this->snakePoses.back().y - this->pommePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y > this->snakePoses.back().y) inputs.push_back((this->pommePoses.back().y - this->snakePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (vectDiff.x == vectDiff.y && vectDiff.x < 0) {
				inputs.push_back((vectDiff.y < 0) ? vectDiff.y / 10.0 : -1.0);
				inputs.push_back((vectDiff.y > 0) ? vectDiff.y / 10.0 : -1.0);
			}
			else {
				inputs.push_back(-1.0);
				inputs.push_back(-1.0);
			}

			// Mur & queue inputs
			for (unsigned i = this->snakePoses.size() - 2; i > this->snakePoses.size() - 1 - this->sizes.back(); i--) {
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x < this->snakePoses.back().x) {
					double dist = (this->snakePoses.back().x - this->snakePoses[i].x) / 10.0;
					if (dist < minDistFace)
						minDistFace = dist;
				}
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y < this->snakePoses.back().y) {
					double dist = (this->snakePoses.back().y - this->snakePoses[i].y) / 10.0;
					if (dist < minDistDroite)
						minDistDroite = dist;
				}
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y > this->snakePoses.back().y) {
					double dist = (this->snakePoses[i].y - this->snakePoses.back().y) / 10.0;
					if (dist < minDistGauche)
						minDistGauche = dist;
				}
			}
			inputs.push_back((minDistFace != 1000) ? minDistFace : this->snakePoses.back().x / 10.0);
			inputs.push_back((minDistDroite != 1000) ? minDistDroite : this->snakePoses.back().y / 10.0);
			inputs.push_back((minDistGauche != 1000) ? minDistGauche : ((SIZE_COTE - 1) - this->snakePoses.back().y) / 10.0);
			break;

		case DROITE:
			// Pomme inputs
			if (this->snakePoses.back().y == this->pommePoses.back().y && this->pommePoses.back().x > this->snakePoses.back().x) inputs.push_back((this->pommePoses.back().x - this->snakePoses.back().x) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y > this->snakePoses.back().y) inputs.push_back((this->pommePoses.back().y - this->snakePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (this->snakePoses.back().x == this->pommePoses.back().x && this->pommePoses.back().y < this->snakePoses.back().y) inputs.push_back((this->snakePoses.back().y - this->pommePoses.back().y) / 10.0);
			else inputs.push_back(-1.0);
			if (vectDiff.x == vectDiff.y && vectDiff.x > 0) {
				inputs.push_back((vectDiff.y > 0) ? vectDiff.y / 10.0 : -1.0);
				inputs.push_back((vectDiff.y < 0) ? vectDiff.y / 10.0 : -1.0);
			}
			else {
				inputs.push_back(-1.0);
				inputs.push_back(-1.0);
			}

			// Mur & queue inputs
			for (unsigned i = this->snakePoses.size() - 2; i > this->snakePoses.size() - 1 - this->sizes.back(); i--) {
				if (this->snakePoses.back().y == this->snakePoses[i].y && this->snakePoses[i].x > this->snakePoses.back().x) {
					double dist = (this->snakePoses[i].x - this->snakePoses.back().x) / 10.0;
					if (dist < minDistFace)
						minDistFace = dist;
				}
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y > this->snakePoses.back().y) {
					double dist = (this->snakePoses[i].y - this->snakePoses.back().y) / 10.0;
					if (dist < minDistDroite)
						minDistDroite = dist;
				}
				if (this->snakePoses.back().x == this->snakePoses[i].x && this->snakePoses[i].y < this->snakePoses.back().y) {
					double dist = (this->snakePoses.back().y - this->snakePoses[i].y) / 10.0;
					if (dist < minDistGauche)
						minDistGauche = dist;
				}
			}
			inputs.push_back((minDistFace != 1000) ? minDistFace : ((SIZE_COTE - 1) - this->snakePoses.back().x) / 10.0);
			inputs.push_back((minDistDroite != 1000) ? minDistDroite : ((SIZE_COTE - 1) - this->snakePoses.back().y) / 10.0);
			inputs.push_back((minDistGauche != 1000) ? minDistGauche : this->snakePoses.back().y / 10.0);
			break;
		}

		vector<double> outputs = this->getResults(inputs);

		// S�curit�
		if (outputs.size() != 3) {
			cout << "Erreur : La taille des \202l\202ments de sortie est erron\202e." << endl;
			exit(-1);
		}

		int output = max_element(outputs.begin(), outputs.end()) - outputs.begin();

		if (output != 0) {
			switch (this->snakeDir) {
			case HAUT:
				if (output == 1) this->snakeDir = DROITE;
				if (output == 2) this->snakeDir = GAUCHE;
				break;

			case BAS:
				if (output == 1) this->snakeDir = GAUCHE;
				if (output == 2) this->snakeDir = DROITE;
				break;

			case GAUCHE:
				if (output == 1) this->snakeDir = HAUT;
				if (output == 2) this->snakeDir = BAS;
				break;

			case DROITE:
				if (output == 1) this->snakeDir = BAS;
				if (output == 2) this->snakeDir = HAUT;
				break;
			}
		}

		if (this->snakeDir == HAUT)
			this->snakePoses.push_back(this->snakePoses.back() - sf::Vector2u(0, 1));
		else if (this->snakeDir == BAS)
			this->snakePoses.push_back(this->snakePoses.back() + sf::Vector2u(0, 1));
		else if (this->snakeDir == GAUCHE)
			this->snakePoses.push_back(this->snakePoses.back() - sf::Vector2u(1, 0));
		else if (this->snakeDir == DROITE)
			this->snakePoses.push_back(this->snakePoses.back() + sf::Vector2u(1, 0));

		this->movesUntilNoAppleEaten++;

		// Se mord la queue
		for (unsigned i = this->snakePoses.size() - 2; i > this->snakePoses.size() - 1 - this->sizes.back(); i--)
			if (this->snakePoses.back() == this->snakePoses[i]) {
				this->score -= 1000;
				this->isAlive = false;
			}

		// Touche un mur
		if (this->snakePoses.back().x >= SIZE_COTE || this->snakePoses.back().y >= SIZE_COTE) {				// Pas besoin de tester les < 0 puisque snakePos est un unsigned
			this->snakePoses.back().x = 0;
			this->snakePoses.back().y = 0;
			this->score -= 100;
			this->isAlive = false;
		}

		// Trop de mouvements sans manger de pomme
		if (this->movesUntilNoAppleEaten > movesLimitUntilNoAppleEaten) {
			this->score -= 500;
			this->isAlive = false;
		}

		// Mange une pomme
		if (this->snakePoses.back() == this->pommePoses.back()) {
			this->score += 200;
			this->movesUntilNoAppleEaten = 0;
			this->sizes.push_back(this->sizes.back() + 1);
			// a fini le jeu
			if (this->sizes.back() == SIZE_COTE * SIZE_COTE) {
				this->isAlive = false;
				this->pommePoses.push_back(sf::Vector2u(-1, -1));
				return;
			}
			sf::Vector2u newPommePos;
			do {
				newPommePos = sf::Vector2u(rand() % SIZE_COTE, rand() % SIZE_COTE);
			} while (find(this->snakePoses.end() - this->sizes.back(), this->snakePoses.end(), newPommePos) != this->snakePoses.end());
			this->pommePoses.push_back(newPommePos);
		}
		else {
			this->sizes.push_back(this->sizes.back());
			this->pommePoses.push_back(this->pommePoses.back());
		}
	}
}

vector<double> operator*(const vector<double>& layer, const vector<vector<double>>& weightMatrice)
{
	vector<double> result;

	for (unsigned matLine = 0; matLine < weightMatrice.size(); matLine++) {
		// S�curit�
		if (weightMatrice[matLine].size() != layer.size()) {
			cout << "Erreur : Produit matriciel impossible." << endl;
			exit(-1);
		}

		double sum = 0.0;
		for (unsigned i = 0; i < layer.size(); i++)
			sum += layer[i] * weightMatrice[matLine][i];

		result.push_back(sum);
	}

	return result;
}

vector<double> operator+(const vector<double>& layer, const vector<double>& biases)
{
	// S�curit�
	if (layer.size() != biases.size()) {
		cout << "Erreur : Somme matricielle impossible." << endl;
		exit(-1);
	}

	vector<double> result;

	for (unsigned i = 0; i < layer.size(); i++)
		result.push_back(layer[i] + biases[i]);

	return result;
}
