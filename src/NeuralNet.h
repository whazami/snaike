#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <SFML/Graphics.hpp>

using namespace std;

constexpr unsigned SIZE_COTE = 13;

class NeuralNet
{
public:
	NeuralNet(vector<unsigned> topology);
	NeuralNet(NeuralNet& net, double randomWeightsBiasesPercentage = -1.0);
	NeuralNet(string fileName = "Best Net.txt");
	vector<double> getResults(vector<double> inputs); 
	void update(unsigned int movesLimitUntilNoAppleEaten);

	// Gestion sauvegarde
	void save(string fileName = "Best Net.txt");

	// Variables propres au jeu
	int score;
	vector<sf::Vector2u> snakePoses, pommePoses;
	vector<unsigned> sizes;
	bool isAlive;

private:
	vector<vector<vector<double>>> weightMats;
	vector<vector<double>> biases;

	static double activation(double x, bool derivative = false);

	vector<unsigned> topology;

	// Variables propres au jeu
	enum { HAUT, BAS, GAUCHE, DROITE } snakeDir;
	unsigned int movesUntilNoAppleEaten;
};

vector<double> operator*(const vector<double>& layer, const vector<vector<double>>& weightMatrice);
vector<double> operator+(const vector<double>& layer, const vector<double>& biases);

#endif // NEURALNET_H