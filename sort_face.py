import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import cv2 # Still useful for reading images if needed, but DeepFace handles its own loading
import paths

# --- Configuration ---
model_name = 'Facenet512' # You can also try 'VGG-Face', 'OpenFace', 'DeepFace', etc.

INPUT_DIR = paths.INPUT_DIR # Folder containing the original photos
last_folder_name = os.path.basename(INPUT_DIR)
OUTPUT_DIR = f"results/{last_folder_name}_{model_name}" # Folder to save the final results

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of valid image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

# --- Data Structure to Hold All Face Information ---
# We store all face data for later sorting: (embedding, image_path)
face_data = []

def get_all_face_data(directory):
    """
    Scans the directory, detects ALL faces in ALL images, and generates embeddings.
    
    Crucial: DeepFace.represent() returns a list of dictionaries for ALL faces detected in an image.
    We need to flatten this into a single list of (embedding, path) pairs.
    """
    
    print("🚀 Starting face detection and embedding generation...")
    
    filenames = os.listdir(directory)
    print(f'---Total files {len(filenames)}---')
    
    for idx, filename in enumerate(filenames):
        print(idx, end='')
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
            
        path = os.path.join(directory, filename)
        
        try:
            # Use DeepFace.represent to get embeddings. 
            # 'ArcFace' is a powerful model. You can also try 'Facenet'.
            # 'detector_backend' uses OpenCV's DNN by default, which is fast and reliable.
            results = DeepFace.represent(
                img_path=path, 
                model_name=model_name, 
                detector_backend="yolov12n",
                enforce_detection=True # Set to False only if you know every image has a face
            )
            
            # Each 'results' item is a dictionary for ONE detected face in the image
            for face_obj in results:
                embedding = face_obj['embedding']
                face_data.append({
                    'embedding': embedding,
                    'image_path': path
                })
            
            print(f"  -> Processed {filename} Found {len(results)} face(s).")
            
        except ValueError as e:
            # DeepFace raises ValueError if no face is detected
            if "Face could not be detected" in str(e):
                print(f"  -> No face detected in {filename}. Skipping.")
            else:
                print(f"  -> Error processing {filename}: {e}")
        except Exception as e:
             print(f"  -> Unexpected error processing {filename}: {e}")

    return face_data


def cluster_and_sort_photos(all_face_data):
    """
    Clusters the embeddings and sorts the photos into named folders.
    """
    
    if not all_face_data:
        print("\n🚫 No faces found to cluster. Check your 'input_photos' folder.")
        return

    # Extract only the embedding vectors into a NumPy array
    embeddings = np.array([d['embedding'] for d in all_face_data])
    
    print(f"\nClustering {len(embeddings)} total face embeddings...")

    # --- DBSCAN Clustering ---
    # eps: The max distance for two embeddings to be considered the same person.
    # ArcFace (used here) has 512-dimensional embeddings, so the distance is larger 
    # than the 128-dim vectors from Dlib/face_recognition.
    # You might need to tune this value (try 1.0 to 1.5)
    EPSILON = 0.7 
    
    cl = DBSCAN(metric="euclidean", n_jobs=-1, eps=EPSILON).fit(embeddings)
    label_ids = cl.labels_
    
    # Get all unique cluster labels/IDs
    unique_labels = np.unique(label_ids)
    
    # 2. Sorting and Saving
    print("\n📦 Sorting photos into person folders...")
    
    for label in unique_labels:
        # Check if the label is the noise cluster (-1)
        is_noise = (label == -1)
        folder_name = "Unknown_Faces" if is_noise else f"Person_{label}"
        person_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Find the indices corresponding to this cluster ID
        indices = np.where(label_ids == label)[0]
        
        # Track the unique image paths that belong to this person/cluster
        images_for_this_person = set()
        for i in indices:
            images_for_this_person.add(all_face_data[i]['image_path'])
            
        print(f"  -> Folder '{folder_name}' contains {len(images_for_this_person)} unique photos.")
        
        # Copy the original image files to the new folder
        for src_path in images_for_this_person:
            # We copy the original photo, not the face crop
            dst_path = os.path.join(person_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            
    print("\n✅ Separation complete!")
    print(f"Results are saved in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    # 1. Load data and generate encodings
    all_face_data = get_all_face_data(INPUT_DIR)
    
    # 2. Cluster and sort
    cluster_and_sort_photos(all_face_data)