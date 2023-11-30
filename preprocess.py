```python
import cv2
import os
import numpy as np

class Preprocess:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load_image(self):
        self.image = cv2.imread(self.input_path)
        if self.image is None:
            print("Unable to load image")
            return False
        return True

    def crop_image(self, x, y, width, height):
        self.image = self.image[y:y+height, x:x+width]

    def filter_image(self):
        # Apply a Gaussian blur
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)

    def save_image(self):
        cv2.imwrite(os.path.join(self.output_path, 'preprocessed.jpg'), self.image)

if __name__ == "__main__":
    input_path = "/path/to/input/image"
    output_path = "/path/to/output/image"
    preprocess = Preprocess(input_path, output_path)
    if preprocess.load_image():
        preprocess.crop_image(100, 100, 200, 200)  # Crop parameters: x, y, width, height
        preprocess.filter_image()
        preprocess.save_image()
```
