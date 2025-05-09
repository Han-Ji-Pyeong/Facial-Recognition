import cv2
import mediapipe as mp
import json
import os
import numpy as np
from datetime import datetime
import traceback

class TroubleshootFacialRecognitionSystem:
    def __init__(self):
        print("Initializing facial recognition system...")
        
        # Create directory for storing face images if it doesn't exist
        if not os.path.exists('faces'):
            os.makedirs('faces')
            print("Created 'faces' directory")
        
        # Initialize or load the database
        self.db_file = 'face_database.json'
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.database = json.load(f)
                print(f"Loaded database with {len(self.database.get('people', []))} people")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.database = {"people": []}
        else:
            print("No existing database found, creating new one")
            self.database = {"people": []}
            self._save_database()
        
        # Initialize MediaPipe Face Detection
        print("Initializing MediaPipe components...")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize video capture
        self.cap = None
        
        print("Initialization complete")
            
    def _save_database(self):
        """Save the database to a JSON file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.database, f, indent=4)
            print(f"Database saved successfully with {len(self.database.get('people', []))} people")
        except Exception as e:
            print(f"Error saving database: {e}")
            
    def add_person(self):
        """Capture a face and add it to the database with information"""
        print("\n=== Adding New Person ===")
        name = input("Enter person's name: ")
        
        # Check if person already exists
        for person in self.database["people"]:
            if person["name"] == name:
                print(f"Person with name '{name}' already exists!")
                return
        
        # Additional information
        info = {}
        info["age"] = input("Enter age: ")
        info["occupation"] = input("Enter occupation: ")
        info["notes"] = input("Enter additional notes: ")
        
        # Capture face
        print("\nPreparing to capture face. Please look at the camera...")
        print("Press 'c' to capture when ready, or 'q' to cancel.")
        
        try:
            print("Opening webcam...")
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("ERROR: Could not open webcam. Please check your camera connection.")
                return
                
            print("Webcam opened successfully")
            face_image = None
            
            with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
                print("Face detection initialized")
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame from webcam")
                        break
                    
                    # Convert the BGR image to RGB and process it with MediaPipe Face Detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    # Draw face detections
                    if results.detections:
                        for detection in results.detections:
                            self.mp_drawing.draw_detection(frame, detection)
                    
                    cv2.imshow('Capture Face', frame)
                    print("Displaying webcam feed...", end='\r')
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nCancelled face capture")
                        break
                    elif key == ord('c'):
                        if not results.detections:
                            print("No face detected! Please try again.")
                            continue
                        elif len(results.detections) > 1:
                            print("Multiple faces detected! Please ensure only one face is in the frame.")
                            continue
                        
                        # Get bounding box
                        detection = results.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                    int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        # Ensure coordinates are within image bounds
                        x, y = max(0, x), max(0, y)
                        w = min(w, iw - x)
                        h = min(h, ih - y)
                        
                        # Extract face region with some margin
                        margin = 20
                        x_with_margin = max(0, x - margin)
                        y_with_margin = max(0, y - margin)
                        w_with_margin = min(w + 2*margin, iw - x_with_margin)
                        h_with_margin = min(h + 2*margin, ih - y_with_margin)
                        
                        face_image = frame[y_with_margin:y_with_margin+h_with_margin, 
                                          x_with_margin:x_with_margin+w_with_margin]
                        
                        # Save the face image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = f"faces/{name}_{timestamp}.jpg"
                        cv2.imwrite(image_filename, face_image)
                        
                        print(f"Face captured successfully and saved as {image_filename}")
                        break
            
            self.cap.release()
            cv2.destroyAllWindows()
            
            if face_image is not None:
                # Add to database
                person_data = {
                    "name": name,
                    "info": info,
                    "image_path": image_filename,
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.database["people"].append(person_data)
                self._save_database()
                print(f"\nSuccessfully added {name} to the database!")
            else:
                print("Failed to capture face. Please try again.")
                
        except Exception as e:
            print(f"Error during face capture: {e}")
            traceback.print_exc()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
    
    def recognize_faces(self):
        """Start the recognition process using webcam"""
        print("\n=== Face Recognition Mode ===")
        print("Press 'q' to quit recognition mode")
        
        if not self.database["people"]:
            print("No people in the database yet. Please add some faces first.")
            return
        
        # Load face images for comparison
        print("Loading face images from database...")
        known_face_images = []
        known_face_names = []
        known_face_info = []
        
        for person in self.database["people"]:
            if os.path.exists(person["image_path"]):
                try:
                    img = cv2.imread(person["image_path"])
                    if img is None:
                        print(f"Warning: Could not load image {person['image_path']}")
                        continue
                    known_face_images.append(img)
                    known_face_names.append(person["name"])
                    known_face_info.append(person["info"])
                    print(f"Loaded image for {person['name']}")
                except Exception as e:
                    print(f"Error loading image {person['image_path']}: {e}")
        
        if not known_face_images:
            print("No valid face images found in the database.")
            return
        
        print(f"Loaded {len(known_face_images)} face images for comparison")
        
        try:
            print("Opening webcam...")
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("ERROR: Could not open webcam. Please check your camera connection.")
                return
                
            print("Webcam opened successfully")
            
            with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
                print("Face detection initialized")
                
                frame_count = 0
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame from webcam")
                        break
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # Print status every 30 frames
                        print(f"Processing frame {frame_count}...", end='\r')
                    
                    # Convert the BGR image to RGB and process it with MediaPipe Face Detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        for detection in results.detections:
                            # Draw the detection
                            self.mp_drawing.draw_detection(frame, detection)
                            
                            # Get bounding box
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                        int(bboxC.width * iw), int(bboxC.height * ih)
                            
                            # Ensure coordinates are within image bounds
                            x, y = max(0, x), max(0, y)
                            w = min(w, iw - x)
                            h = min(h, ih - y)
                            
                            # Extract face region
                            face_img = frame[y:y+h, x:x+w]
                            
                            if face_img.size == 0:
                                continue
                                
                            # Simple face recognition by comparing with stored faces
                            best_match_index = -1
                            best_match_score = float('inf')
                            
                            # Resize for comparison
                            face_img_resized = cv2.resize(face_img, (100, 100))
                            face_gray = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)
                            
                            for i, known_face in enumerate(known_face_images):
                                try:
                                    # Resize known face for comparison
                                    known_face_resized = cv2.resize(known_face, (100, 100))
                                    
                                    # Convert to grayscale for comparison
                                    known_gray = cv2.cvtColor(known_face_resized, cv2.COLOR_BGR2GRAY)
                                    
                                    # Calculate mean squared error as a simple similarity measure
                                    err = np.sum((face_gray.astype("float") - known_gray.astype("float")) ** 2)
                                    err /= float(face_gray.shape[0] * face_gray.shape[1])
                                    
                                    if err < best_match_score:
                                        best_match_score = err
                                        best_match_index = i
                                except Exception as e:
                                    print(f"Error comparing with face {i}: {e}")
                            
                            # Draw rectangle around face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Display name and info if a match is found
                            threshold = 1500  # Adjust this threshold as needed
                            if best_match_index != -1 and best_match_score < threshold:
                                name = known_face_names[best_match_index]
                                info = known_face_info[best_match_index]
                                
                                # Display name and score
                                name_text = f"{name} ({best_match_score:.1f})"
                                cv2.putText(frame, name_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Display additional info
                                y_position = y + h + 20
                                for key, value in info.items():
                                    info_text = f"{key}: {value}"
                                    cv2.putText(frame, info_text, (x, y_position), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    y_position += 20
                            else:
                                unknown_text = f"Unknown ({best_match_score:.1f})"
                                cv2.putText(frame, unknown_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # Add status text
                    status_text = f"Press 'q' to quit | {len(known_face_names)} faces in database"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display the resulting image
                    cv2.imshow('Face Recognition', frame)
                    
                    # Hit 'q' on the keyboard to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting face recognition mode")
                        break
            
            self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during face recognition: {e}")
            traceback.print_exc()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
    
    def list_people(self):
        """List all people in the database"""
        print("\n=== People in Database ===")
        if not self.database["people"]:
            print("No people in the database yet.")
            return
            
        for i, person in enumerate(self.database["people"], 1):
            print(f"{i}. {person['name']}")
            print(f"   Added on: {person['added_date']}")
            print(f"   Image: {person['image_path']}")
            print("   Information:")
            for key, value in person["info"].items():
                print(f"      {key}: {value}")
            print()
    
    def delete_person(self):
        """Delete a person from the database"""
        self.list_people()
        
        if not self.database["people"]:
            return
            
        try:
            index = int(input("\nEnter the number of the person to delete (0 to cancel): ")) - 1
            if index == -1:
                print("Operation cancelled")
                return
                
            if 0 <= index < len(self.database["people"]):
                person = self.database["people"][index]
                name = person["name"]
                
                # Delete the image file if it exists
                if os.path.exists(person["image_path"]):
                    try:
                        os.remove(person["image_path"])
                        print(f"Deleted image file: {person['image_path']}")
                    except Exception as e:
                        print(f"Error deleting image file: {e}")
                    
                # Remove from database
                self.database["people"].pop(index)
                self._save_database()
                print(f"\nSuccessfully deleted {name} from the database!")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a valid number")
    
    def update_person_info(self):
        """Update information for a person"""
        self.list_people()
        
        if not self.database["people"]:
            return
            
        try:
            index = int(input("\nEnter the number of the person to update (0 to cancel): ")) - 1
            if index == -1:
                print("Operation cancelled")
                return
                
            if 0 <= index < len(self.database["people"]):
                person = self.database["people"][index]
                name = person["name"]
                
                print(f"\nUpdating information for {name}")
                print("Leave field empty to keep current value")
                
                # Display current info
                print("\nCurrent information:")
                for key, value in person["info"].items():
                    print(f"  {key}: {value}")
                
                # Update info
                new_info = {}
                for key in person["info"]:
                    new_value = input(f"Enter new {key} (current: {person['info'][key]}): ")
                    new_info[key] = new_value if new_value else person["info"][key]
                
                # Add new field if desired
                if input("\nAdd new information field? (y/n): ").lower() == 'y':
                    while True:
                        new_key = input("Enter new field name (or empty to stop): ")
                        if not new_key:
                            break
                        new_value = input(f"Enter value for {new_key}: ")
                        new_info[new_key] = new_value
                
                # Update database
                person["info"] = new_info
                self._save_database()
                print(f"\nSuccessfully updated information for {name}!")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a valid number")
    
    def run(self):
        """Run the main application loop"""
        while True:
            print("\n=== Facial Recognition System ===")
            print("1. Add a new person")
            print("2. Recognize faces")
            print("3. List all people")
            print("4. Delete a person")
            print("5. Update person information")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                self.add_person()
            elif choice == '2':
                self.recognize_faces()
            elif choice == '3':
                self.list_people()
            elif choice == '4':
                self.delete_person()
            elif choice == '5':
                self.update_person_info()
            elif choice == '6':
                print("Exiting the application. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

# Run the application
if __name__ == "__main__":
    print("Starting Facial Recognition System...")
    app = TroubleshootFacialRecognitionSystem()
    app.run()