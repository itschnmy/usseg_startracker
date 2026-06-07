import cv2
from detector import StarDetector

# load image
image = cv2.imread("7b51bada-3192-4e1d-8757-edf577d06e89.jfif", cv2.IMREAD_GRAYSCALE)

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