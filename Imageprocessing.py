import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\pc\Downloads\dark-skies-peagus2-web.webp"

# Load image
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Estimate background bằng blur mạnh
background = cv2.GaussianBlur(blur, (51, 51), 0)

# Trừ nền
clean = cv2.subtract(blur, background)

mean = np.mean(clean)
std = np.std(clean)

k = 4  # có thể chỉnh 3–5
_, binary = cv2.threshold(
    clean,
    mean + k * std,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")

plt.subplot(232)
plt.imshow(blur, cmap='gray')
plt.title("Gaussian Blur")

plt.subplot(233)
plt.imshow(background, cmap='gray')
plt.title("Background")

plt.subplot(234)
plt.imshow(clean, cmap='gray')
plt.title("Background Subtracted")

plt.subplot(235)
plt.imshow(binary, cmap='gray')
plt.title("Threshold")


for ax in plt.gcf().axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

