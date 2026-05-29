import cv2
from detector import StarDetector

def detector(image_name, sigma_threshold, min_area):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_name}")

    star_detector = StarDetector(sigma_threshold, min_area)
    stars = star_detector.process(image)

    # Sort brightest stars
    stars = sorted(stars, key=lambda s: s.intensity, reverse=True)

    # keep only top N stars to reduce false detections, comment out to skip this step
    max_stars = 6
    stars = stars[:max_stars]

    print("Number of stars:", len(stars))

    return stars