import cv2

def visualizer(img):
    centroids = []
    with open("centroids.txt", "r") as f:
        for line in f:
            x, y = map(float, line.split())
            centroids.append((x, y))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, (x, y) in enumerate(centroids):
        x = int(round(x))
        y = int(round(y))
        cv2.circle(img, (x, y), 8, (0, 255, 0), 1)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imwrite("marked.png", img)
    print("Saved marked.png")