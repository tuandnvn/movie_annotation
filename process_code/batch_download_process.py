import os, sys

import argparse
from multiprocessing import Process, Pool, Queue
from process import dlfile
import cv2

from vgg_19_keras import VGG_19

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
def get_links(stop_counter, url_queue ):
	counter = 0
	for download_urls in download_urlss:
		with open(download_urls, 'r') as fh:
			for line in fh:
				# put one more url into the queue, block if it is full
				url_queue.put( line.strip(), block = True, timeout = None)
				counter += 1
				if stop_counter <= counter:
					return

'''
'''
def download( url_queue, downloaded_queue ):
	while True:
		url = url_queue.get(block = True, timeout = None)
		path = os.path.join(temp_directory, url.split("/")[-1])
		print 'Download url %s to %s' % (url, path)
		dlfile(url, username, password, path, "wb", protected_url = PROTECTED_URL)

		downloaded_queue.put( path, block = True, timeout = None)

mean_pix = [103.939, 116.779, 123.68]
def get_features_from_path (cgg_model, sample_ratio, path):
	def process_frame ( frame ):
		# Subtract mean of color fields for each color
		for i in xrange(3):
			frame[:,:,i] -= mean_pix[i]

		# ( height, width, channels ) -> ( channels, height, width )
		frame = frame.transpose((2,0,1))

		return frame

	cap = cv2.VideoCapture(input)

	counter = 0

	sequence_features = []

	while (cap.isOpened()):
		ret, frame = cap.read()

		if not ret:
			break 


		if counter % sample_ratio == 0:
			height, width, channels = frame.shape

			# Reduce the size of the frame before cropping
			fix_size = 256
			cropped_size = 224

			if height < width:
				frame = cv2.resize(frame, dsize = (fix_size, 0), interpolation = cv2.INTER_AREA)
			else:
				frame = cv2.resize(frame, dsize = (0, fix_size), interpolation = cv2.INTER_AREA)

			# Subtract the mean
			frame = process_frame(frame)

			# Prepare a numpy array to be passed to the CGG network
			imgs = np.zeros(10, 3, cropped_size, cropped_size)

			# oversample (4 corners, center, and their x-axis flips)

			indices_height = [0, height - cropped_size + 1]
			indices_width = [0, width - cropped_size + 1]
			
			center_height = height / 2 - cropped_size + 1
			center_width = width / 2 - cropped_size + 1

			curr = 0
			for i in indices_height:
				for j in indices_width:
					imgs[curr, :, :, :] = frame[:, i: i + cropped_size, j: j + cropped_size]
					curr += 1
			imgs[4, :, :, :] = frame[:, center_height: center_height + cropped_size, center_width : center_width + cropped_size]

			for i in xrange(5):
				# Mirror image of the first five
				imgs[i + 5, :, :, :] = imgs[i, ::-1 , :, :]

			features = cgg_model.predict(imgs)

			avg_features = np.mean(features, axis = 0)
			
			sequence_features.append(avg_features)


		counter += 1

	# Before   # of frames x np.array[ 3, cropped_size, cropped_size ]
	# After     np.array[ # of frames, 3, cropped_size, cropped_size ]
	sequence_features = np.stack ( sequence_features, axis = 0 )

	return sequence_features

'''
'''

def process ( cgg_model, sample_ratio, feature_directory, downloaded_queue, processed_queue ):
	while True:
		path = downloaded_queue.get(block = True, timeout = None)

		# Get filename from path
		fn = path.rsplit(os.path.sep)[-1]

		fv_filename = fn[:fn.rfind('.')] + '.fv' 

		# Make output feature path
		output_path = os.path.join(feature_directory, fn)

		sequence_features = get_features_from_path ( cgg_model, sample_ratio, path )

		np.save(output_path, sequence_features)

		processed_queue.put(path, block = True, timeout = None)

# Run the following
# python batch_download_process.py -m ../models/vgg19_weights.h5 -f ../features -t ../tmp -s 100 -p 4 -r 10
if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='A script to download and process movie snippets in all training/validating/testing dataset into their appropriate output')

	parser.add_argument('-f', '--feature_directory', action='store', metavar = ('OUTPUR_DIR'),
								help = "The output directory to store the feature vectors. For each clip, a feature vector file will be generated, " )

	parser.add_argument('-t', '--temp_directory', action='store', metavar = ('TEMP_DIR'), 
								help = "Temporary directory to store movie files which might be removed immediately" )
	
	parser.add_argument('-s', '--stop_counter', action='store', metavar = ('STOP_COUNTER'), 
								help = "Stop process after processing a number of clips" )

	parser.add_argument('-p', '--no_process', action='store', metavar = ('NUMBER_OF_PROCESSES'), 
								help = "Number of parallel process" )

	parser.add_argument('-m', '--model', action='store', metavar = ('MODEL'), 
								help = "Where you store the model file" )

	parser.add_argument('-r', '--sample_ratio', action='store', metavar = ('SAMPLE_RATIO'), 
								help = "Sample n th frame of each video clip" )

	parser.add_argument('--removetemp', action='store_true', dest='remove_temp',
                    help='Remove temporary clip immediately')

	username = "5Dvs"
	password = "ma8Aofazah"

	args = parser.parse_args()

	temp_directory = args.temp_directory
	feature_directory = args.feature_directory
	stop_counter = int(args.stop_counter)
	no_process = int(args.no_process)
	remove_temp = args.remove_temp
	sample_ratio = int(args.sample_ratio)

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
	get_link_process = Process(target=get_links, args = (stop_counter, link_q))
	get_link_process.start()
    

    # Queue that stores paths of files that has been downloaded
	downloaded_q = Queue(maxsize = 1000)

	for i in range(no_process):
		Process(target=download, args=(link_q, downloaded_q)).start()


    # Queue that stores files that has been processed to be removed later on
    # If remove_temp == True, we would not keep more than 1000 clip files
	processed_q = Queue(maxsize = 1000)

	for i in range(no_process):
		Process(target=process, args=(model, sample_ratio, feature_directory, downloaded_q, processed_q)).start()

	get_link_process.join()

