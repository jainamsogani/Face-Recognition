# Read and show video stream, capture images

import cv2
import numpy as np

# Initialize Camera
cap = cv2.VideoCapture(0)

# Face Detection
# Change this location to the location where Python is installed in your PC
location = "C:\\Users\\jainj\\AppData\\Local\\Programs\\Python\\Python39\\"
face_cascade = cv2.CascadeClassifier(location + "Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = '.\\data\\'
file_name = input("Enter the name of the person : ")

while True:
	ret, frame = cap.read()

	if ret == False:
		continue

	# Detect faces and show bounding box
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	if len(faces) == 0:
		continue

	# Sort the faces according to the area (f[2]*f[3])
	faces = sorted(faces, key = lambda f:f[2]*f[3], reverse = True)

	for face in faces:
		x, y, w, h = face
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		skip += 1
		if skip % 10 == 0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Frame", frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Converting the face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save the data into file system
np.save(dataset_path + file_name + '.npy', face_data)
print("Data Successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()