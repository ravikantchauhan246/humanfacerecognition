import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from io import BytesIO
import base64

# Install required packages
try:
    from mtcnn import MTCNN
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mtcnn", "tensorflow"])
    from mtcnn import MTCNN

# Function to load an image from a URL
def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    return Image.open(BytesIO(image_data))

# Function to detect faces in an image
def detect_faces(image):
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create MTCNN detector
    detector = MTCNN()
    
    # Detect faces
    faces = detector.detect_faces(image)
    return faces, image

# Function to draw bounding boxes around detected faces
def draw_faces(image, faces):
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display the image
    ax.imshow(image)
    
    # Draw rectangles around faces
    for face in faces:
        x, y, width, height = face['box']
        rect = plt.Rectangle((x, y), width, height, 
                             fill=False, color='green', linewidth=2)
        ax.add_patch(rect)
        
        # Draw keypoints
        for key, point in face['keypoints'].items():
            ax.plot(point[0], point[1], 'r.')
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print confidence scores
    for i, face in enumerate(faces):
        print(f"Face {i+1} confidence: {face['confidence']:.4f}")

# Example usage with a sample image
def main():
    # Sample image URL (you can replace with your own)
    image_url = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg"
    
    print("Loading image...")
    image = load_image_from_url(image_url)
    
    print("Detecting faces...")
    faces, image_array = detect_faces(image)
    
    print(f"Found {len(faces)} faces!")
    draw_faces(image_array, faces)

if __name__ == "__main__":
    main()