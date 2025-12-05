#pragma once
#include <string>
#include <vector>

struct Pair {
    double id1; //id of catalog star 1
    double id2; // id of catalog star 2
    double cosTheta; //angular distance between 2 stars, cos(theta12) = the dot product of 2 unit vectors of 2 stars
};

class PairCatalog {
private:
    std::vector<Pair> pairs;
    auto queryPairs(double cosLow, double cosHigh); //find pairs whose cos(Theta) value in range [low, high] of the detected pair
public:
    bool loadFile(std::string path); //load precomputed data from a csv. file and store them in a vector. return false if errors occur, otherwise return true
    std::vector<Pair> getPairs();
};