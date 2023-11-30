```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, input_path, output_path, annotation_path):
        self.input_path = input_path
        self.output_path = output_path
        self.annotation_path = annotation_path

    def load_images(self):
        self.images = []
        for filename in os.listdir(self.input_path):
            img = cv2.imread(os.path.join(self.input_path, filename))
            if img is not None:
                self.images.append(img)
        return self.images

    def load_annotations(self):
        with open(self.annotation_path, 'r') as file:
            self.annotations = file.readlines()
        return self.annotations

    def serialize_annotations(self):
        self.serialized_annotations = []
        for annotation in self.annotations:
            serialized_annotation = self.serialize_annotation(annotation)
            self.serialized_annotations.append(serialized_annotation)
        return self.serialized_annotations

    def serialize_annotation(self, annotation):
        # Implement this method based on your annotation format
        pass

    def split_dataset(self, test_size=0.2):
        self.images_train, self.images_test, self.annotations_train, self.annotations_test = train_test_split(
            self.images, self.serialized_annotations, test_size=test_size, random_state=42)

    def save_dataset(self):
        np.save(os.path.join(self.output_path, 'images_train.npy'), self.images_train)
        np.save(os.path.join(self.output_path, 'images_test.npy'), self.images_test)
        np.save(os.path.join(self.output_path, 'annotations_train.npy'), self.annotations_train)
        np.save(os.path.join(self.output_path, 'annotations_test.npy'), self.annotations_test)

if __name__ == "__main__":
    input_path = "/path/to/input/images"
    output_path = "/path/to/output/dataset"
    annotation_path = "/path/to/input/annotations.txt"
    dataset = Dataset(input_path, output_path, annotation_path)
    dataset.load_images()
    dataset.load_annotations()
    dataset.serialize_annotations()
    dataset.split_dataset()
    dataset.save_dataset()
```
