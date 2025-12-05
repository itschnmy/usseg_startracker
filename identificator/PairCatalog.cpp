#include "PairCatalog.h"
#include <iostream>
#include <fstream>

bool PairCatalog::loadFile(std::string path) {}

std::vector<Pair> PairCatalog::getPairs() {
    return pairs;
}

auto PairCatalog::queryPairs(double low, double high) {
    auto iterationLow = std::lower_bound(
        pairs.begin(), pairs.end(), low, [](Pair p, double v){
            return p.cosTheta < v;
        } 
    );

    auto iterationHigh = std::upper_bound(
        pairs.begin(), pairs.end(), high, [](double v, const Pair p){
            return v < p.cosTheta;
        }
    );

    return {iterationLow, iterationHigh};
}