import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5 
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  
video=cv2.VideoCapture(0) 

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)


print('Processing unknown faces...')
counter = 0
skip_frames = 5  
while True:
    ret, image = video.read()
    if not ret:
        break
    
    counter += 1
    if counter % skip_frames != 0:
        continue

    image = cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1])))

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    
    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

    
        match = 'Unknown'
        if True in results:  
            match = known_names[results.index(True)]
            print(f'match found - {match} ')

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0,255,0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)
    
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cv2.destroyAllWindows()
video.release()