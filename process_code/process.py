import os
from scipy import ndimage
from subprocess import check_output

import cv2
import numpy as numpy

import urllib2
from urllib2 import urlopen, URLError, HTTPError, Request
import argparse

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='A script to download/process movie snippets')

	username = "5Dvs"
	password = "ma8Aofazah"

	# Download file should be put on the same directory as the process code
	# If couldn't find, download it then
	all_urls = "downloadLinksAvi.txt"

	# For downloading movie snippets from the website
	parser.add_argument('download', nargs = 2, help = "Download snippets of a movie, such as [American_Beauty]\
	 							into a directory [output_dir]" )
	parser.add_argument('sampling', nargs = 3, help = "Sample from snippets from a movie dir [input_dir]\
	 							into a directory [output_dir], [sample ratio]" )

	args = parser.parse_args()

	if len(args.download) == 2:
		movie_name = args.download[0]
		output_dir = args.download[1]

		if not os.path.exist(download_file):
			all_urls_url = "http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/downloadLinksAvi.txt"

			# Using this code to download
			# http://stackoverflow.com/questions/4028697/how-do-i-download-a-zip-file-in-python-using-urllib2
			dlfile(all_urls_url, username, password, all_urls, "w")

		if os.path.exist(output_dir):
			print 'Directory exists. Please remove it first.'
			break

		# Create directory
		os.makedirs(output_dir)

		with open(all_urls, "r") as all_urls_file:
			for url in all_urls_file:
				if movie_name in url:
					path = os.path.join( output_dir, os.path.basename(url) )
					dlfile(url, username, password, path, "wb")

	if len(args.sampling) == 2:
		input_dir = args.download[0]
		output_dir = args.download[1]
		sample_ratio = args.download[2]

		if os.path.exist(output_dir):
			print 'Directory exists. Please remove it first.'
			break

		# Create directory
		os.makedirs(output_dir)

		for avi_file in glob.glob( os.path.join( input_dir, "*.avi" )):
			sub_output_dir = os.path.join(output_dir, os.path.basename(avi_file)[:-len(".avi")])
			# Create sub output dir
			os.makedirs(sub_output_dir)
			sample_avi(avi_file, sample_ratio, sub_output_dir)


def dlfile(url, username, password, output, write_code):
    # Open the url
    try:
    	if username == "":
        	f = urlopen(url)
        else:
        	data = { 'username': username,'password': password }
        	req = Request(url, data)
        	f = urlopen(req)

        print "-- Downloading " + url

        # Open our local file for writing
        with open(output, write_code) as local_file:
            local_file.write(f.read())

    #handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url

def sample_avi(avi_file, sample_ratio, output_dir):
	cap = cv2.VideoCapture(avi_file)

	counter = 0
	while (cap.isOpened()):
		ret, frame = cap.read()

		if not ret:
			return 

		if counter % sample_ratio == 0:
			path = os.path.join(output_dir, counter + '.png')
			cv2.imwrite(path, frame)
		counter += 1



