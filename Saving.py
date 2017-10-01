import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()

	if ret == True:
		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	 	
		if key == ord(' '):
	 		cv2.imwrite("must.jpg", frame)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
cv2.destroyAllWindows()