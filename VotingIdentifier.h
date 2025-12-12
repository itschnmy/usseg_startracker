#pragma once
#include "StarCatalog.h"
#include "CatalogStar.h"
#include <Eigen/Dense>
#include <vector>

struct Pair { //structure for a precomputed catalog star pair including theirs ids and the angular distance between these stars
    double id1; // id of catalog star 1
    double id2; // id of catalog star 2
    double cosTheta; // cos(theta12), = the dot product of 2 unit vectors of star catalog 1 & 2 
};

class VotingIdentifier {
private:
    StarCatalog& cat; // input catalog
    double angleTol; //the angular tolerance threshold in radians
    std::vector<Pair> pairs //the list of precomputed star pairs

    static double clampDot(double d); //***to make sure dot product is between -1 -> 1
    void buildPairs(double fov, double magLimit);
    CatalogStar* findStar(int id);
    auto queryPairs(double low, double high);
public:
    
};