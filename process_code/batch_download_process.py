import os, sys

import argparse
from multiprocessing import Process, Pool, Queue
from process import dlfile

PROTECTED_URL = 'http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected'

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='A script to download and process movie snippets in all training/validating/testing dataset into their appropriate output')

	parser.add_argument('-f', '--feature-directory', nargs = 1, metavar = ('OUTPUR_DIR'), 
								help = "The output directory to store the feature vectors. For each clip, a feature vector file will be generated, " )

	parser.add_argument('-t', '--temp-directory', nargs = 1, metavar = ('TEMP_DIR'), 
								help = "Temporary directory to store movie files which might be removed immediately" )
	
	parser.add_argument('-s', '--stop-counter', nargs = 1, metavar = ('STOP_COUNTER'), 
								help = "Stop process after processing a number of clips" )

	username = "5Dvs"
	password = "ma8Aofazah"

	args = parser.parse_args()

	temp_directory = arg.temp-directory
	feature_directory = arg.feature-directory


	training_urls = "../LSMDC task/LSMDC16_annos_training.csv"	
	val_urls = "../LSMDC task/LSMDC16_annos_val.csv"
	test_urls = "../LSMDC task/LSMDC16_annos_test.csv"
	blindtest_urls = "../LSMDC task/LSMDC16_annos_blindtest.csv"

	download_urlss = ["../LSMDC task/MPIIMD_downloadLinks.txt", "../LSMDC task/MVADaligned_downloadLinks.txt", "../LSMDC task/BlindTest_downloadLinks.txt"]

	# Queue that stores the url 
	# Here we use a thread-safe queue 
	link_q = Queue(maxsize = 1000)

	def get_links( q ):
		for download_urls in download_urlss:
			with open(download_urls, 'r') as fh:
				for line in fh:
					# put one more url into the queue, block if it is full
					q.put( line.strip(), block = True, timeout = None)

	# Just start reading from the url files
	get_link_process = Process(target=get_links, args = (link_q, ))
    get_link_process.start()
    
    def download( q ):
    	while True:
    		url = q.get(True)
    		path = os.path.join(temp_directory, )
    		print 'Download url %s to %s' % (url, path)
    		dlfile(url, username, password, path, "wb", protected_url = PROTECTED_URL)

    download_pool = Pool(5, download, ( link_q, ))


    get_link_process.join()

