#pragma once
#include "CatalogStar.h"
#include <vector>
#include <string>

class StarCatalog {
private:
    std::vector<CatalogStar> stars;
public:
    bool loadFile(std::string path); //load data from csv and push into a vector. return false if errors occur, otherwise return true
    std::vector<CatalogStar> getStars(); //return a star pair
};