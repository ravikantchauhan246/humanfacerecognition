# Real-time Face Recognition

This project provides a real-time face recognition system using your webcam. It can detect faces and match them against known faces stored in a directory.

## Setup Instructions

1. Install the required packages:
```
pip install opencv-python mtcnn tensorflow scikit-learn pillow
```

2. Add known face images to the `known_faces` directory:
   - The system will create this directory automatically on first run if it doesn't exist
   - Add clear, frontal face images of people you want to recognize
   - Name each file with the person's name (e.g., `john.jpg`, `sarah.png`) - the filename (without extension) will be used as the identity label

3. Run the real-time detection script:
```
python real_time_detection.py
```

## Usage

- The program will access your webcam and start detecting faces
- Detected faces will be highlighted with a colored rectangle:
  - **Green**: Face matched with a known person (name will be displayed)
  - **Red**: Face not matched with any known person
- Press 'q' to quit the application

## How It Works

The system:
1. Loads and processes reference images from the `known_faces` directory
2. Extracts facial embeddings (feature vectors) using VGG16 pre-trained on ImageNet
3. Captures webcam frames and detects faces using MTCNN
4. Compares detected faces with known faces using cosine similarity
5. Displays real-time results showing whether a face is matched or not

## Files

- `real_time_detection.py` - The main script for real-time face recognition
- `face_recognition.py` - An example script for face detection from an image URL

## Note

This system uses a basic face recognition approach with VGG16 feature extraction. For production use, consider more specialized face recognition models like FaceNet or ArcFace. 