import cv2
from detector import StarDetector

# load image
image = cv2.imread("1_clean_gemini.png", cv2.IMREAD_GRAYSCALE)

# create detector
detector = StarDetector()

# detect stars
stars = detector.process(image)

# extract centroid list
centroids = [star.position for star in stars]

print("Number of stars:", len(centroids))

for c in centroids:
    print(c)

with open("centroids.txt", "w") as f:
    for star in stars:
        x, y = star.position
        f.write(f"{x} {y}\n")