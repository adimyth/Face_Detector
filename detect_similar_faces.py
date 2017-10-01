import dlib
import scipy.misc
import numpy as np
import os
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import glob
import time

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
# This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.
face_recognition_model = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
TOLERANCE = 0.5

def get_face_encodings(path):
	image = scipy.misc.imread(path)
	# number of faces to detect
	detected_faces = face_detector(image,1)
	shapes_faces = [shape_predictor(image,face) for face in detected_faces]
	# Get face_encodings for faces
	return [np.array(face_recognition_model.compute_face_descriptor(image,face_pose,1)) for face_pose in shapes_faces]

def get_face_encodings_camera(image):
	detected_faces = face_detector(image,1)		
	shapes_faces = [shape_predictor(image,face) for face in detected_faces]
	return [np.array(face_recognition_model.compute_face_descriptor(image,face_pose,1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces,face):
	return (np.linalg.norm(known_faces - face,axis = 1) <= TOLERANCE)

def find_match(known_faces,names,face):
	matches = compare_face_encodings(known_faces,face)
	count = 0
	for match in matches:
		if match:
			return names[count]
		count += 1
	return "Intruder"

image_filenames = filter(lambda x: x.endswith('.jpg'),os.listdir('images/'))
image_filenames = sorted(image_filenames)
names = [x[:-4] for x in image_filenames]
print("These are my training Images :")
for img in image_filenames:
	print(img)
paths_to_images = ['images/'+x for x in image_filenames]  
face_encodings  = []
for img in paths_to_images:
	encoding = get_face_encodings(img)
	if len(encoding) == 0:
		print("No face found in the image")
		exit()
	elif len(encoding) > 1:
		print("More then 1 face found in the image")
		exit()
	face_encodings.append(get_face_encodings(img)[0])

# WebCam Check
print("Warming up camera...")
vs = WebcamVideoStream(src = 0).start()
while True:
	img = vs.read()
	cv2.imshow("Frame",img)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	encoding = get_face_encodings_camera(img)
	if len(encoding) == 0:
		print("No face found in the image")
		# Allow the person to make movements
		time.sleep(20)
	match = find_match(face_encodings,names,encoding[0])
	print(match)

# File Check
image_filenames = filter(lambda x: x.endswith('.jpg'),os.listdir('test/'))
print("These are my testing Images :")
for img in image_filenames:
	print(img)
paths_to_images = ['test/'+x for x in image_filenames]

print("Checking whether test images exist in Training set")
for img in paths_to_images:
	encoding = get_face_encodings(img)
	if len(encoding) == 0:
		print("No face found in the image")
		exit()
	elif len(encoding) > 1:
		print("More then 1 face found in the image")
		exit()
	match = find_match(face_encodings,names,encoding[0])
	print(img,match)
