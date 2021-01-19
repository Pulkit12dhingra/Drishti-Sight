#flask model
from flask import Flask, render_template, Response
import numpy as np
import argparse
import time
import cv2
import os
#import speech_recognition as sr
from gtts import gTTS
import playsound

app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join(["C:/Users/pulki/Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS-main", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join(["C:/Users/pulki/Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS-main", "yolov3.weights"])
configPath = os.path.sep.join(["C:/Users/pulki/Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS-main", "yolov3.cfg"])


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
cap = cv2.VideoCapture(0)
mydict={"earlier":""}
def gen():

	while True:
		ret,image = cap.read()
		(H, W) = image.shape[:2]
		

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()


		print("[INFO] YOLO took {:.6f} seconds".format(end - start))


		boxes = []
		confidences = []
		classIDs = []
		ID = 0


		for output in layerOutputs:

			for detection in output:

				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]


				if confidence > args["confidence"]:

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")




					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))


					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)


		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])


		if len(idxs) > 0:
			list1 = []
			for i in idxs.flatten():

				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				centerx = round((2*x + w)/2)
				centery = round((2*y + h)/2)
				if centerX <= W/3:
					W_pos = "left "
				elif centerX <= (W/3 * 2):
					W_pos = "center "
				else:
					W_pos = "right "

				if centerY <= H/3:
					H_pos = "top "
				elif centerY <= (H/3 * 2):
					H_pos = "mid "
				else:
					H_pos = "bottom "
				print(x,y,w,h)
				print("area is = ",((x+w)*(y+h)))
				area=((x+w)*(y+h))
				if area > 200000:
					distance_measure="really close to you"
				elif area<200000 and area>100000:
					distance_measure="close to you"
				else:
					distance_measure="far from you"
				

			list1.append("there is a "+ LABELS[classIDs[i]] + " in "+ H_pos + W_pos  + distance_measure)
			description = ', '.join(list1)
			
			if description!=mydict["earlier"]:
				myobj = gTTS(text=description, lang="en", slow=False)
				try:
					myobj.save("object_detection.mp3")
				except:
					pass
				playsound.playsound("object_detection.mp3", True)
				#print(description,earlier,"not")
				mydict["earlier"]=description
			else:
				#print(description,earlier,"equals")
				pass
		#cv2.imshow("Frame",image)
		ret, jpeg = cv2.imencode('.jpg', image)
		image= jpeg.tobytes()
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port


    app.run(host='0.0.0.0',port='5000', debug=True)