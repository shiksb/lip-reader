import numpy as np
import matplotlib
matplotlib.use('PS')
import os
import dlib # run "pip install dlib"
import cv2 # run "pip install opencv-python"
import imageio

from scipy import misc # run "pip install pillow"
from imutils import face_utils



#http://www.scipy-lectures.org/advanced/image_processing/
#http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/


RECTANGLE_LENGTH = 90

					
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


def crop_and_save_image(img, img_path, img_name):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	# load the input image, resize it, and convert it to grayscale

	image = cv2.imread('dataset/' + img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if len(rects) > 1:
		print( "ERROR: more than one face detected")
		return
	if len(rects) < 1:
		print( "ERROR: no faces detected")
		return 

	rect = rects[0]
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	w = RECTANGLE_LENGTH
	h = RECTANGLE_LENGTH
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	(x_r, y_r, w_r, h_r) = (x, y, w, h)
 
	# show the face number
	cv2.putText(image, "Face #{}".format(0 + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	crop_img = image[y_r:y_r + h_r, x_r:x_r + w_r]
	print( '/cropped/' + img_path)
	cv2.imwrite('cropped/' + img_path, crop_img)


people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
data_types = ['phrases', 'words']
words = ['01','02','03','04','05','06','07','08','09','10',]


if not os.path.exists('cropped'):
	os.mkdir('cropped')

for person_ID in people:
	if not os.path.exists('cropped/' + person_ID ):
			os.mkdir('cropped/' + person_ID)
	for data_type in data_types:
		if not os.path.exists('cropped/' + person_ID + '/' + data_type):
			os.mkdir('cropped/' + person_ID + '/' + data_type)

		for phrase_ID in words:
			if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID):
				os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID)

			for instance_ID in words:
				directory = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'

				print('Created: dataset/' + directory)
				filelist = os.listdir('dataset/' + directory)
				if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID):
					os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID)

					for img_name in filelist:
						if img_name.startswith('color'):
							image = misc.imread('dataset/' + directory + '' + img_name)

							crop_and_save_image(image, directory + '' + img_name, img_name)


max_width = 0
max_height = 0
min_width = 10000
min_height = 10000

for trial in people:
	for word_index, word in enumerate(words):
		for i in words:
			path = os.path.join('cropped', trial, 'words', word, i)
			files = os.listdir(path + '/')
			for img_name in files:
				if img_name.startswith('color'):
					img = imageio.imread(path + '/' + img_name)
					w,h,c = np.shape(img)
					if w > max_width: max_width = w
					if w < min_width: min_width = w
					if h > max_height: max_height = h
					if h < min_height: min_height = h
	print('Completed scanning person #' + trial)

print('max height: ' + str(max_height))
print('max width: ' + str(max_width))
print('min height: ' + str(min_height))
print('min width: ' + str(min_width))