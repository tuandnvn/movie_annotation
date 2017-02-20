import os, sys

import argparse
from multiprocessing import Process, Pool, Queue, TimeoutError
import glob
import time

import cv2
import numpy as np

import tensorflow as tf
from vgg_19_keras import VGG_19
from process import dlfile

PROTECTED_URL = 'http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected'

# Remove the last two layers of VGG to get features only
# The last two layers are dropout and softmax
def VGG_features(weights_path=None):
	model = VGG_19(weights_path)
	model.pop()
	model.pop()
	return model

'''
'''
def get_links(url_queue, fromLine, toLine ):
	if toLine == -1:
		toLine = sys.maxint
	counter = 0
	for download_urls in download_urlss:
		with open(download_urls, 'r') as fh:
			for line in fh:
				counter += 1
				if counter >= fromLine:
					# put one more url into the queue, block if it is full
					url_queue.put( line.strip(), block = True, timeout = None)
				if counter >= toLine:
					return

'''
'''
def download( url_queue, downloaded_queue, processed_files, downloaded_files ):
	while True:
		url = url_queue.get(block = True, timeout = None)
		avi_file = url.split("/")[-1]
		avi_file_without_avi = avi_file[:-len('.avi')]
		if avi_file_without_avi in processed_files:
			continue

		path = os.path.join(temp_directory, avi_file)

		if not avi_file_without_avi in downloaded_files:
			print 'Download url %s to %s' % (url, path)
			dlfile(url, username, password, path, "wb", protected_url = PROTECTED_URL)

		downloaded_queue.put( path, block = True, timeout = None)

mean_pix = [103.939, 116.779, 123.68]
def get_features_from_path (device, cgg_model, sample_ratio, path):
	def process_frame ( frame ):
		# Subtract mean of color fields for each color
		for i in xrange(3):
			frame[:,:,i] -= mean_pix[i]

		# ( height, width, channels ) -> ( channels, height, width )
		frame = frame.transpose((2,0,1))

		return frame

	t = time.time()

	cap = cv2.VideoCapture(path)

	counter = 0

	sequence_features = []
	sequence_imgs = []

	while (cap.isOpened()):
		ret, frame = cap.read()

		if not ret:
			break

		if counter % sample_ratio == 0:
			
			# Change to float type
			frame = frame.astype(np.float32)

			height, width, channels = frame.shape

			# Reduce the size of the frame before cropping
			fix_size = 256
			cropped_size = 224

			if height < width:
				# dsize = ( # of columns, # of rows ) = ( width, height )
				frame = cv2.resize(frame, dsize = (fix_size * width / height, fix_size), interpolation = cv2.INTER_AREA)
			else:
				frame = cv2.resize(frame, dsize = (fix_size, fix_size * height / width), interpolation = cv2.INTER_AREA)


			# Subtract the mean
			# and transfer to ( channels, height, width )
			frame = process_frame(frame)


			# Get size after reshape
			_, height, width = frame.shape


			# Prepare a numpy array to be passed to the CGG network
			imgs = np.zeros( (10, 3, cropped_size, cropped_size) )

			# oversample (4 corners, center, and their x-axis flips)

			indices_height = [0, height - cropped_size]
			indices_width = [0, width - cropped_size]
			
			center_height = (height - cropped_size) / 2
			center_width = (width - cropped_size) / 2

			curr = 0
			for i in indices_height:
				for j in indices_width:
					imgs[curr, :, :, :] = frame[:, i: i + cropped_size, j: j + cropped_size]
					curr += 1
			imgs[4, :, :, :] = frame[:, center_height: center_height + cropped_size, center_width : center_width + cropped_size]

			for i in xrange(5):
				# Mirror image of the first five
				imgs[i + 5, :, :, :] = imgs[i, ::-1 , :, :]

			sequence_imgs.append(imgs)

		counter += 1

	print 'Read content and do simple processing uses %.2f seconds' % (time.time() - t)
	t = time.time()

	if len(sequence_imgs) == 0:
		return None

	# np.array[ # of frames x 10, 3, cropped_size, cropped_size ]
	sequence_imgs = np.concatenate ( sequence_imgs, axis = 0 )

	# print 'sequence_imgs shape = ', sequence_imgs.shape

	# np.array[ # of frames x 10, 4096 ]
	with tf.device(device):
		sequence_features = cgg_model.predict(sequence_imgs)

	# np.array[ # of frames, 10, 4096 ]
	sequence_features = np.reshape(sequence_features, (sequence_features.shape[0] // 10, 10, -1))

	# Average over 10 images
	sequence_features = np.mean(sequence_features, axis = 1)

	# print 'sequence_features shape = ', sequence_features.shape
	

	# if len ( sequence_features ) != 0:
	# 	# Before   # of frames x np.array[ 3, cropped_size, cropped_size ]
	# 	# After     np.array[ # of frames, 3, cropped_size, cropped_size ]
	# 	sequence_features = np.stack ( sequence_features, axis = 0 )

	print 'Run the data through the CNN network %.2f seconds' % (time.time() - t)

	return sequence_features

'''
Parameters:
device: Choose a device to process (GPU or CPU)
cgg_model: 

'''
def process ( device, cgg_model, sample_ratio, feature_directory, downloaded_queue, processed_queue ):
	try:
		while True:
			path = downloaded_queue.get(block = True, timeout = 30)

			# Get filename from path
			fn = path.rsplit(os.path.sep)[-1]

			fv_filename = fn[:fn.rfind('.')] + '.fv' 

			# Make output feature path
			output_path = os.path.join(feature_directory, fv_filename)

			t = time.time()
			sequence_features = get_features_from_path ( device, cgg_model, sample_ratio, path )

			if sequence_features != None:
				np.save(output_path, sequence_features)

				processed_queue.put(path, block = True, timeout = None)

				processed_time = time.time() - t

				processed_time_per_frame = processed_time / sequence_features.shape[0]

				print 'Process from video %s to feature path % s using %.2f seconds, per frame = %.2f seconds' % (path, output_path, processed_time, processed_time_per_frame)
	except TimeoutError, Queue.Empty:
		print 'Stop processing when there is no more files to process'

def remove_file( processed_queue ):
	while True:
		path = processed_queue.get(block = True, timeout = None)

		os.remove(path)

		print 'Remove file %s' % path

# Run the following
# python batch_download_process.py -m ../models/vgg19_weights.h5 -f ../feature_vectors -t ../tmp --from 0 --to 10 -p 4 -r 10 -d /gpu:0 --removetemp
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='A script to download and process movie snippets in all training/validating/testing dataset into their appropriate output')

	parser.add_argument('-f', '--feature_directory', action='store', metavar = ('OUTPUR_DIR'),
								help = "The output directory to store the feature vectors. For each clip, a feature vector file will be generated, " )

	parser.add_argument('-t', '--temp_directory', action='store', metavar = ('TEMP_DIR'), 
								help = "Temporary directory to store movie files which might be removed immediately" )

	parser.add_argument('-p', '--no_process', action='store', metavar = ('NUMBER_OF_PROCESSES'), 
								help = "Number of parallel process" )

	parser.add_argument('-m', '--model', action='store', metavar = ('MODEL'), 
								help = "Where you store the model file" )

	parser.add_argument('-r', '--sample_ratio', action='store', dest = 'sample_ratio', 
								help = "Sample n th frame of each video clip" )

	parser.add_argument('-d', '--device', action='store', dest = 'device', 
								help = "Select device to run predicting code (should be gpu if you have one)" )

	parser.add_argument('--from', action='store', dest = 'fromLine', 
								help = "Start processing from line ?" , default = '0')

	parser.add_argument('--to', action='store', dest = 'toLine', 
								help = "Finish processing to line exclusively? -1 is no upper limit" , default = '-1')

	parser.add_argument('--replace-features', action='store_true', dest = 'replace_features', 
								help = "If feature vector file exists, replace features. Default is false" )

	parser.add_argument('--removetemp', action='store_true', dest='remove_temp',
                    help='Remove temporary clip immediately. Default is false')

	username = "5Dvs"
	password = "ma8Aofazah"

	args = parser.parse_args()

	temp_directory = args.temp_directory
	feature_directory = args.feature_directory
	no_process = int(args.no_process)
	remove_temp = args.remove_temp
	sample_ratio = int(args.sample_ratio)
	device = args.device
	fromLine = int(args.fromLine)
	toLine = int(args.toLine)
	replace_features = args.replace_features

	print 'replace_features %s' % replace_features
	print 'remove_temp %s' % remove_temp


	# Store whether a file name has been processed
	processed_files = {}
	downloaded_files = {}

	# If keep feature vectors in feature_directory 
	if not replace_features:
		feature_files = glob.glob(os.path.join(feature_directory, '*.fv.npy') )
		for feature_file in feature_files:
			ff = feature_file.split("/")[-1]
			processed_files[ff[:- len('.fv.npy')]] = True

		dl_files = glob.glob(os.path.join(temp_directory, '*.avi') )
		for dl_file in dl_files:
			ff = dl_file.split("/")[-1]
			downloaded_files[ff[:- len('.avi')]] = True


	with tf.device(device):
		# Load on only one device
		model = VGG_features(args.model)

	training_urls = "../LSMDC task/LSMDC16_annos_training.csv"	
	val_urls = "../LSMDC task/LSMDC16_annos_val.csv"
	test_urls = "../LSMDC task/LSMDC16_annos_test.csv"
	blindtest_urls = "../LSMDC task/LSMDC16_annos_blindtest.csv"

	download_urlss = ["../LSMDC task/MPIIMD_downloadLinks.txt", "../LSMDC task/MVADaligned_downloadLinks.txt", "../LSMDC task/BlindTest_downloadLinks.txt"]

	# Queue that stores the url 
	# Here we use a thread-safe queue 
	link_q = Queue(maxsize = 1000)
	

	# Just start reading from the url files
	get_link_process = Process(target=get_links, args = (link_q, fromLine, toLine))
	get_link_process.start()
    

    # Queue that stores paths of files that has been downloaded
	downloaded_q = Queue(maxsize = 1000)

	for i in range(no_process):
		Process(target=download, args=(link_q, downloaded_q, processed_files, downloaded_files)).start()


    # Queue that stores files that has been processed to be removed later on
    # If remove_temp == True, we would not keep more than 1000 clip files
	processed_q = Queue(maxsize = 1000)

	# processes = []
	# for i in range(1):
	# 	p = Process(target=process, args=('/gpu:%d' % i, model, sample_ratio, feature_directory, downloaded_q, processed_q))
	# 	processes.append(p)
	# 	p.start()

	if remove_temp:
		remove_process = Process(target=remove_file, args=(processed_q, )).start()

	process(device, model, sample_ratio, feature_directory, downloaded_q, processed_q)

	get_link_process.join()

	# for p in processes:
	# 	p.join()
	# while True:
	# 	processed_path = processed_q.get(block = True, timeout = None)
	# 	print 'Process %s' % processed_path