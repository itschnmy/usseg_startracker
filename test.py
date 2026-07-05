# import image_processing
from plateSolver import Tetra3
from detector import StarDetector
from visualize_centroids import visualizer

import cv2
import sys
import numpy as np

"""
Input Script: python3 main.py [img] [sigma_threshold] [min_area] [hfov] [pic_height] [pic_width]
Eg: py main.py 7b51bada-3192-4e1d-8757-edf577d06e89.jfif 8 7 70
"""

def main():
    # Command arguments
    img = str(sys.argv[1])
    sigma = float(sys.argv[2])
    min_area = float(sys.argv[3])
    hfov = float(sys.argv[4])
    if len(sys.argv) != 5:
        print("At least a parameter missed!!!")
        sys.exit()


    # Read image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]


    # Centroiding
    detector = StarDetector(sigma, min_area)
    stars = detector.process(image)
    centroids = np.array([star.position for star in stars])

    print("Number of stars:", len(centroids))
    """for c in centroids:
        print(c)""" #uncomment to print out the list of centroids

    with open("centroids.txt", "w") as f:
        for star in stars:
            x, y = star.position
            f.write(f"{x} {y}\n")


    # Visualize centroids for further centroiding analysis
    visualizer(image)


    # Plate solving
    size = (height, width) 
    t3 = Tetra3(load_database="default_database")
    result = t3.solve_from_centroids(
        star_centroids=centroids,
        size=size,
        fov_estimate=hfov,
        fov_max_error=10,
        pattern_checking_stars=21,
        match_radius=0.03,
        match_threshold=1e-2
    )

    print(t3.database_properties)
    print(t3.has_database)
    print("centroid shape:", centroids.shape)
    print("first 10:", centroids[:10])

    for key, value in result.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()