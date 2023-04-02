#include "NetManager.h"

#include <iostream>
#include <SFML/Graphics.hpp>

using namespace std;

sf::RectangleShape pixels[SIZE_COTE][SIZE_COTE];

int main(void)
{
	srand((unsigned)time(NULL));
	sf::RenderWindow window(sf::VideoMode(793, 793), "SFML Window");

	for (int i = 0; i < SIZE_COTE; i++)
		for (int j = 0; j < SIZE_COTE; j++) {
			pixels[i][j] = sf::RectangleShape(sf::Vector2f(window.getSize().x / SIZE_COTE - 1, window.getSize().y / SIZE_COTE - 1));
			pixels[i][j].setOrigin(pixels[i][j].getLocalBounds().width / 2.f, pixels[i][j].getLocalBounds().height / 2.f);
			pixels[i][j].setPosition(window.getSize().x / SIZE_COTE * (j + 1.f / 2.f), window.getSize().y / SIZE_COTE * (i + 1.f / 2.f));
			pixels[i][j].setFillColor(sf::Color::Black);
		}

	int t = 0;
	sf::Clock clk;

	NetManager netManager({ 8, 5, 3 }, 40); 
	NeuralNet* bestNet = NULL;
	unsigned generation = 0;

	cout << "G\202n\202ration 0" << endl;

	while (window.isOpen())
	{
		sf::Event e;
		while (window.pollEvent(e))
		{
			if (e.type == sf::Event::Closed)
				window.close();

			if (e.type == sf::Event::Resized)
				window.setView(sf::View(sf::FloatRect(0.f, 0.f, e.size.width, e.size.height)));

			if (e.type == sf::Event::KeyPressed)
				if (e.key.code == sf::Keyboard::Space) {
					generation++;
					t = 0;
					system("CLS");
					cout << "G\202n\202ration " << generation << endl;
					for (int i = 0; i < SIZE_COTE; i++)
						for (int j = 0; j < SIZE_COTE; j++)
							pixels[i][j].setFillColor(sf::Color::Black);
				}

			if (e.type == sf::Event::KeyPressed)
				if (e.key.code == sf::Keyboard::S)
					if (bestNet != NULL) {
						bestNet->save();
						cout << "R\202seau de neurones sauvegard\202 !" << endl;
					}
		}

		// Update Nets
		netManager.update();
		
		// Update Best Net's game
		if (clk.getElapsedTime().asMilliseconds() > 50) {
			bestNet = netManager.getBestNet(generation);
			if (bestNet != NULL) {
				// Update pixels
				vector<sf::Vector2u> snakePoses = bestNet->snakePoses;
				vector<sf::Vector2u> pommePoses = bestNet->pommePoses;
				vector<unsigned> sizes = bestNet->sizes;
				if (t >= sizes[t])
					pixels[snakePoses[t - sizes[t]].y][snakePoses[t - sizes[t]].x].setFillColor(sf::Color::Black);
				pixels[snakePoses[t].y][snakePoses[t].x].setFillColor(sf::Color::White);
				pixels[pommePoses[t].y][pommePoses[t].x].setFillColor(sf::Color::Red);

				if (t < snakePoses.size() - 1) t++;
				else {
					generation++;
					t = 0;
					system("CLS");
					cout << "G\202n\202ration " << generation << endl;
					for (int i = 0; i < SIZE_COTE; i++)
						for (int j = 0; j < SIZE_COTE; j++)
							pixels[i][j].setFillColor(sf::Color::Black);
				}
			}
			clk.restart();
		}

		window.clear();
		for (int i = 0; i < SIZE_COTE; i++)
			for (int j = 0; j < SIZE_COTE; j++)
				window.draw(pixels[i][j]);
		window.display();
	}

	return 0;
}
