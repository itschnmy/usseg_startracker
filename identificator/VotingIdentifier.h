#pragma once
#include "PairCatalog.h"
#include "StarCatalog.h"

class VotingIdentifier {
private:
    StarCatalog& scat; //input star catalog (base catalog)
    PairCatalog& pcat; //input pair catalog (catalog with precomputed angular distance)
    double angleTol; //angle tolerence threshold in rad, how close 2 angles must be to be considered a match
public:
    VotingIdentifier(StarCatalog& scat, PairCatalog& pscat, double angleTol);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> identify(std::vector<DetectedStar> detected);
    //using voting method, compare the detected stars with the star catalog to identify them
};