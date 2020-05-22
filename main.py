import numpy as np
import argparse
import time
import cv2
from scipy.spatial import distance as dist
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--model", type=str, 
	default='SSD',
	help="SSD or YOLO?")
ap.add_argument("--input", type=str,
	default='videos/clip.mp4',
	help="path to optional input video file")
ap.add_argument("--output", type=str,
	default='output/vid.avi',
	help="path to optional output video file")
ap.add_argument("--probability", type=float, 
	default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("--threshold", type=float,
	default=0.4,
	help="max allowed distance between objects")
args = vars(ap.parse_args())

if args['model'] is 'SSD':
	model = 'models/ssd_mobilenet_v1.pb'
	pbtxt = 'models/ssd_mobilenet_v1.pbtxt'
	net = cv2.dnn.readNetFromTensorflow(model, pbtxt)
	ln = None

elif args['model'] is 'YOLO':
	weights = 'models/yolov3.weights'
	config = 'models/yolov3.cfg'
	net = cv2.dnn.readNetFromDarknet(config, weights)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	LABELS = open('models/coco.names').read().strip().split("\n")

# if a video path was not supplied, grab a reference to the webcam
if args['input'] is None:
	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(0)
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

totalframes = 0
starttime = time.time()

NMSthres = 0.7


def detected_objects(frame, net, ln):
	NMS_rects = []
	rects = []
	centroids = []
	confidences = []
	net.setInput(cv2.dnn.blobFromImage(frame, size=(608, 608), swapRB=True, crop=False))
	if args['model'] is 'SSD':
		preds = net.forward()
		preds = np.array(preds)	
		for detection in preds[0, 0]:
			confidence = float(detection[2])
			if confidence > args["probability"] and int(detection[1]) == 1:
				box = detection[3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				width = abs(endX - startX)
				height = abs(endY - startY)
				# draw a black rectangle around detected objects
				cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 0), thickness=2)
				NMS_rects.append([startX, startY, int(width), int(height)])
				rects.append(box.astype("int"))
				centroids.append((int((startX + endX) / 2.0), int((startY + endY) / 2.0)))
				confidences.append(confidence)
		return rects, centroids, confidences, NMS_rects

	elif args['model'] is 'YOLO':
		layer_outputs = net.forward(ln)
		for output in layer_outputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > args['probability'] and LABELS[classID] is 'person':
					print(detection[0:4])
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					startX = int(centerX - (width / 2))
					startY = int(centerY - (height / 2))
					endX = int(centerX + (width / 2))
					endY = int(centerY + (height / 2))
					# draw a black rectangle around detected objects
					cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 0), thickness=2)
					NMS_rects.append([startX, startY, int(width), int(height)])
					rects.append((startX, startY, endX, endY))
					centroids.append((int((startX + endX) / 2.0), int((startY + endY) / 2.0)))
					confidences.append(confidence)
		return rects, centroids, confidences, NMS_rects




# loop over frames from the video stream
while True:

	ret, frame = vs.read()
	if ret is False:
		break
	totalframes += 1

	(H, W) = frame.shape[:2]

	rects = []
	centroids = []
	confidences = []
	NMS_rects = []

	rects, centroids, confidences, NMS_rects = detected_objects(frame, net, ln)
	
	rects = np.array(rects)
	centroids = np.array(centroids)

	# idxs = cv2.dnn.NMSBoxes(NMS_rects, confidences, args['probability'], NMSthres)

	# not using NMS (non-maxima suppression to suppress weak, overlapping bounding)
	idxs = np.arange(len(rects))

	if len(idxs) > 0:
		idxs = idxs.ravel()
		rects = rects[idxs]
		centroids = centroids[idxs]
	
		# distances between the centroids
		distance = dist.cdist(centroids, centroids)
		distance = np.array(distance)

		for i in range(distance.shape[0]):
			for j in range(distance.shape[1]):
				if i >= j:
					distance[i, j] = 9999

		# in order to perform this matching we must (1) find the
		# smallest value in each row and then (2) sort the row
		# indexes based on their minimum values so that the row
		# with the smallest value as at the *front* of the index list
		rows = distance.min(axis=1).argsort()

		# next, we perform a similar process on the columns by
		# finding the smallest value in each column and then
		# sorting using the previously computed row index list
		cols = distance.argmin(axis=1)[rows]

		# minimum distance between two people for social distancing 
		distance_thres = args['threshold'] * (H + W) / 10

		# green boxes around objects closer than distance_thres
		for (row, col) in zip(rows, cols):
			if distance[row, col] < distance_thres:
				print("Violation detected")
				# adding green boxes
				cv2.rectangle(frame, (int(rects[row, 0]), int(rects[row, 1])), \
					(int(rects[row, 2]), int(rects[row, 3])), (0, 255, 0), thickness=2)
				cv2.rectangle(frame, (int(rects[col, 0]), int(rects[col, 1])), \
					(int(rects[col, 2]), int(rects[col, 3])), (0, 255, 0), thickness=2)
	

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
	
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

if args['input'] is None:
	vs.stop()

totaltime = time.time()-starttime
print(totalframes, "total frames", totalframes/totaltime, " FPS")

# close any open windows
cv2.destroyAllWindows()