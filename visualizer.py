import cv2
# star detector's visualization tool
def visualizer(image_name, stars, output_name="marked.png"):
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_name}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i, star in enumerate(stars):
        x, y = star.position
        x = int(round(x))
        y = int(round(y))

        cv2.circle(image, (x, y), 8, (0, 255, 0), 1)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.putText(
            image,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(output_name, image)
    print(f"Saved {output_name}")