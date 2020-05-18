import numpy as np
import argparse
import time
import cv2
from scipy.spatial import distance as dist
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--pbtxt", type=str, 
	default='models/ssd_mobilenet_v1.pbtxt',
	help="path to tf 'deploy' pbtxt file")
ap.add_argument("--model", type=str, 
	default='models/ssd_mobilenet_v1.pb',
	help="path to tf mobile_net pre-trained model")
ap.add_argument("--input", type=str,
	default='videos/clip.mp4',
	help="path to optional input video file")
ap.add_argument("--output", type=str,
	default='output/vid.avi',
	help="path to optional output video file")
ap.add_argument("--probability", type=float, 
	default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("--threshold", type=float,
	default=0.4,
	help="max allowed distance between objects")
args = vars(ap.parse_args())

	
net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])

# if a video path was not supplied, grab a reference to the webcam
if args['input'] is None:
	print("[INFO] starting webcam...")
	vs = cv2.VideoCapture(0)
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

totalframes = -1

starttime = time.time()


def drawbox(frame, objects_detected, flag=False):

	for obj, info in objects_detected.items():
		box = info[0]
		(startX, startY, endX, endY) = box
		confidence = info[1]
		label = '%s: %.2f' % (obj, confidence)

		# no violation detected
		if not flag:
			cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 0), thickness=2)
		# violation detected
		else:
			cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), thickness=2)
			
		cv2.putText(frame, label, (int(startX), int(startY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def obj_detection(frame, model, probability):

	model.setInput(cv2.dnn.blobFromImage(frame, size=(512, 512), swapRB=True, crop=False))
	preds = net.forward()
	preds = np.array(preds)	

	objects_detected = dict()
	trackers_dict = dict()

	(H, W) = frame.shape[:2]

	id_num = 0

	# the values in detection-
	# [batchId, classId, confidence, left, top, right, bottom]
	for detection in preds[0, 0]:
		score = float(detection[2])
		if score > probability and int(detection[1]) == 1:

			box = detection[3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box

			#remove these two lines and use before drawbox
			#rects.append(box)
			#centroids.append((int((startX + endX) / 2.0), int((startY + endY) / 2.0)))

			label = 'person_' + str(id_num)

			while True:
				if label not in objects_detected.keys():
					break
				id_num += 1
				label = 'person_' + str(id_num)

			objects_detected[label] = [tuple(box), score]

	objects_list = list(objects_detected.keys())

	if len(objects_list) > 0:
		trackers_dict = {key : cv2.TrackerKCF_create() for key in (objects_list)}
		for obj in trackers_dict.keys():
			trackers_dict[obj].init(frame, objects_detected[obj][0])
	
	return objects_detected, trackers_dict

objects_detected = dict()

_, frame = vs.read()
objects_detected, trackers_dict =  obj_detection(frame, net, args['probability'])


# loop over frames from the video stream
while True:

	ret, frame = vs.read()
	if ret is False:
		break

	totalframes += 1

	(H, W) = frame.shape[:2]

	rects = []
	centroids = []

	if len(objects_detected) > 0:
		del_items = []
		for obj, tracker in trackers_dict.items():
			ok, bbox = tracker.update(frame)
			if ok:
				objects_detected[obj][0] = bbox
			else:
				del_items.append(obj) 
		for item in del_items:            
			trackers_dict.pop(item)
			objects_detected.pop(item)

		if len(objects_detected) > 0:
			drawbox(frame, objects_detected)
		else:
			objects_detected, trackers_dict =  obj_detection(frame, net, args['probability'])


	'''

	# instead use object tracking
	else:
		print('kcf')
		if len(objects_detected) > 0:

			del_items = []
			for obj, tracker in trackers_dict.items():
				found, box = tracker.update(frame)
				if found:
					(startX, startY, endX, endY) = box
					# draw a black rectangle around detected objects
					drawbox(frame, box, flag=True)

					rects.append(box)
					centroids.append((int((startX + endX) / 2.0), int((startY + endY) / 2.0)))
				
				else:
					del_items.append(obj)

		for item in del_items:
			trackers_dict.pop(item)
			objects_detected.pop(item)


			
	if len(rects) != 0:
		rects = np.array(rects)
		centroids = np.array(centroids)
	
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
		# with the smallest value as at the *front* of the index
		# list
		rows = distance.min(axis=1).argsort()

		# next, we perform a similar process on the columns by
		# finding the smallest value in each column and then
		# sorting using the previously computed row index list
		cols = distance.argmin(axis=1)[rows]

		# minimum distance between two people for social distancing 
		threshold = args['threshold'] * (H + W) / 10

		# green boxes around objects closer than threshold
		for (row, col) in zip(rows, cols):
			if distance[row, col] < threshold:
				# print("Violation detected")
				# adding green boxes
				cv2.rectangle(frame, (int(rects[row, 0]), int(rects[row, 1])), \
					(int(rects[row, 2]), int(rects[row, 3])), (0, 255, 0), thickness=2)
				cv2.rectangle(frame, (int(rects[col, 0]), int(rects[col, 1])), \
					(int(rects[col, 2]), int(rects[col, 3])), (0, 255, 0), thickness=2)

	'''
	

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
