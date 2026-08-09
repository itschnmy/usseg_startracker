import cv2
import numpy as np
import math
from pathlib import Path

class ImagePreprocessor:
    """
    Class executing the advanced image preprocessing pipeline for Star Tracker.
    Aligns with the European Space Agency's (ESA) Tetra3 dynamic calibration pipeline.
    """
    def __init__(self, bg_scale: float = 0.03125, sigma: float = 3.0, filtsize: int = 3):
        self.bg_scale = bg_scale
        self.sigma = sigma
        self.filtsize = filtsize
        
        self.crop_enabled = False
        self.crop_width = 0
        self.crop_height = 0
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        
        self.crop_offset_x_actual = 0
        self.crop_offset_y_actual = 0

        # ESA Tetra3 uses a 3x3 cross structuring element
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def set_crop(self, width: int, height: int, offset_x: int = 0, offset_y: int = 0):
        self.crop_enabled = True
        self.crop_width = width
        self.crop_height = height
        self.crop_offset_x = offset_x
        self.crop_offset_y = offset_y

    def disable_crop(self):
        self.crop_enabled = False
        self.crop_width = 0
        self.crop_height = 0
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.crop_offset_x_actual = 0
        self.crop_offset_y_actual = 0

    def preprocess(self, input_image: np.ndarray):
        """
        Executes the preprocessing pipeline on an input image.
        Returns:
            success (bool): True if preprocessing succeeded
            clean_output (np.ndarray): Background-subtracted image
            binary_output (np.ndarray): Binarized and morphologically opened mask
        """
        if input_image is None or input_image.size == 0:
            return False, None, None

        # 1. Grayscale Conversion (if necessary)
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_image

        # 2. Cropping / ROI Selection
        if self.crop_enabled and self.crop_width > 0 and self.crop_height > 0 and \
           self.crop_width <= gray.shape[1] and self.crop_height <= gray.shape[0]:
            
            offs_y = (gray.shape[0] - self.crop_height) // 2 + self.crop_offset_y
            offs_x = (gray.shape[1] - self.crop_width) // 2 + self.crop_offset_x

            # Clamp offsets to be inside the image boundaries
            offs_y = max(0, min(offs_y, gray.shape[0] - self.crop_height))
            offs_x = max(0, min(offs_x, gray.shape[1] - self.crop_width))

            self.crop_offset_x_actual = offs_x
            self.crop_offset_y_actual = offs_y

            processing_img = gray[offs_y:offs_y+self.crop_height, offs_x:offs_x+self.crop_width]
        else:
            self.crop_offset_x_actual = 0
            self.crop_offset_y_actual = 0
            processing_img = gray

        # 3. Gaussian Blur (3x3 Kernel, Sigma = 1.0)
        blur = cv2.GaussianBlur(processing_img, (3, 3), 1.0)

        # 4. TETRA3 DYNAMIC CALIBRATION STAGE: Median Background Estimation
        small_width = round(processing_img.shape[1] * self.bg_scale)
        small_height = round(processing_img.shape[0] * self.bg_scale)

        ksize = self.filtsize
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 3:
            ksize = 3

        min_size = max(3, ksize)
        small_width = max(small_width, min_size)
        small_height = max(small_height, min_size)

        small_bg = cv2.resize(blur, (small_width, small_height), interpolation=cv2.INTER_NEAREST)
        small_bg = cv2.medianBlur(small_bg, ksize)
        
        background = cv2.resize(small_bg, (processing_img.shape[1], processing_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Perform pixel-wise clipping subtraction
        clean_output = cv2.subtract(blur, background)

        # 5. TETRA3 DYNAMIC CALIBRATION STAGE: RMS Noise Estimation & Thresholding
        l2_norm = cv2.norm(clean_output, cv2.NORM_L2)
        rms_val = l2_norm / math.sqrt(clean_output.size)
        
        threshold_val = self.sigma * rms_val
        if threshold_val > 255.0:
            threshold_val = 255.0

        _, binary_output = cv2.threshold(clean_output, threshold_val, 255, cv2.THRESH_BINARY)

        # 6. Morphological Opening
        binary_output = cv2.morphologyEx(binary_output, cv2.MORPH_OPEN, self.kernel)

        return True, clean_output, binary_output

if __name__ == "__main__":
    # Test script applying the preprocessor
    image_dir = Path("starimage")
    adjust_root = image_dir / "adjust"
    adjust_root.mkdir(exist_ok=True)
    images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.bmp', '.png', '.jpg', '.jpeg'}]
    
    preprocessor = ImagePreprocessor()
    
    for img_path in images:
        print(f"Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        success, clean, binary = preprocessor.preprocess(image)
        if success:
            out_dir = adjust_root / img_path.stem
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / "1_clean.png"), clean)
            cv2.imwrite(str(out_dir / "2_binary.png"), binary)
    print("Done.")