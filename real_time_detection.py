import cv2
import numpy as np
from mtcnn import MTCNN
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image

# Initialize VGG16 model for feature extraction (using pre-trained weights)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def get_face_embedding(face_img):
    # Resize to VGG16 input size
    face_img = cv2.resize(face_img, (224, 224))
    # Convert to PIL Image
    face_img = Image.fromarray(face_img)
    # Convert to array and preprocess
    face_array = np.array(face_img)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)
    # Extract features
    embedding = model.predict(face_array)
    return embedding

def load_known_faces(faces_dir="known_faces"):
    known_face_embeddings = []
    known_face_names = []
    
    # Create directory if it doesn't exist
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        print(f"Created directory: {faces_dir}")
        print("Please add known face images to this directory and run again.")
        return known_face_embeddings, known_face_names
    
    # Load images from the known_faces directory
    for filename in os.listdir(faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract name from filename (without extension)
            name = os.path.splitext(filename)[0]
            
            # Load and process image
            img_path = os.path.join(faces_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
                
            # Convert BGR to RGB (matching MTCNN expected format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face
            detector = MTCNN()
            faces = detector.detect_faces(rgb_image)
            
            if faces:
                # Take the first face found
                x, y, width, height = faces[0]['box']
                face_img = rgb_image[y:y+height, x:x+width]
                
                # Get face embedding
                embedding = get_face_embedding(face_img)
                
                # Store face info
                known_face_embeddings.append(embedding)
                known_face_names.append(name)
                print(f"Loaded known face: {name}")
            else:
                print(f"No face detected in {filename}")
    
    return known_face_embeddings, known_face_names

def realtime_face_detection():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Load known faces
    known_face_embeddings, known_face_names = load_known_faces()
    
    if not known_face_embeddings:
        print("No known faces found. Please add images to the 'known_faces' directory.")
    else:
        print(f"Loaded {len(known_face_embeddings)} known faces")
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert from BGR to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(rgb_frame)
        
        # Draw rectangles around faces
        for face in faces:
            x, y, width, height = face['box']
            
            # Extract the face region
            face_img = rgb_frame[y:y+height, x:x+width]
            
            # Default status and color if no known faces loaded
            status = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            
            # Match face if we have known faces
            if known_face_embeddings and face_img.size > 0:
                try:
                    # Get embedding for current face
                    current_embedding = get_face_embedding(face_img)
                    
                    # Compare with known faces
                    similarities = [cosine_similarity(current_embedding, known_emb)[0][0] 
                                  for known_emb in known_face_embeddings]
                    
                    # Find best match
                    best_match_idx = np.argmax(similarities)
                    best_match_score = similarities[best_match_idx]
                    
                    # Threshold for considering it a match
                    if best_match_score > 0.75:
                        status = f"Match: {known_face_names[best_match_idx]}"
                        color = (0, 255, 0)  # Green for match
                    else:
                        status = f"No Match ({best_match_score:.2f})"
                        color = (0, 0, 255)  # Red for no match
                except Exception as e:
                    print(f"Error matching face: {str(e)}")
                    status = "Error"
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
            
            # Draw status text
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw keypoints
            keypoints = face['keypoints']
            for point in keypoints.values():
                cv2.circle(frame, point, 2, (0, 0, 255), 2)
                
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_face_detection()

print("This is a webcam-based face detection script that would run on your local machine with a webcam.")