#!/usr/bin/python3
import numpy as np
import face_recognition as fr
import cv2
import os 

cwd = os.getcwd()
dir = cwd + '\images'

video_capture = cv2.VideoCapture(0)
encodings = []
names = []

persons = os.listdir(dir)
print(persons)
for person in persons:
    perdir = dir + '\\'+person
    pix = os.listdir(perdir)
    print(pix)
    for img in pix:
        per = fr.load_image_file(img)
        face_enc = fr.face_encodings(per)[0]
        encodings.append(face_enc)
        names.append(person)



while True: 
    ret, frame = video_capture.read()
    

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
