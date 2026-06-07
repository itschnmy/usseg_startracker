import cv2

image = cv2.imread("7b51bada-3192-4e1d-8757-edf577d06e89.jfif", cv2.IMREAD_UNCHANGED)

centroids = []
with open("centroids.txt", "r") as f:
    for line in f:
        x, y = map(float, line.split())
        centroids.append((x, y))

if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for i, (x, y) in enumerate(centroids):
    x = int(round(x))
    y = int(round(y))
    cv2.circle(image, (x, y), 8, (0, 255, 0), 1)
    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imwrite("marked.png", image)
print("Saved marked.png")