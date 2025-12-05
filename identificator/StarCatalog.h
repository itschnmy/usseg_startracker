#pragma once
#include "CatalogStar.h"
#include <vector>
#include <string>

class StarCatalog {
private:
    std::vector<CatalogStar> stars;
public:
    bool loadFile(std::string path); //load the base catalog from a .csv file and store them into a vector. return false if errors occur, otherwise return true
    std::vector<CatalogStar> getStars();
};
