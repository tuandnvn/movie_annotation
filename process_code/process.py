import os, sys, base64
import subprocess, shlex
import urllib2
from urllib2 import urlopen, URLError, HTTPError, Request
import argparse
import glob

import cv2
import numpy as numpy
from scipy import ndimage

# Using this code to download
# http://stackoverflow.com/questions/4028697/how-do-i-download-a-zip-file-in-python-using-urllib2
def dlfile(url, username, password, output, write_code, protected_url = ''):
    # Open the url
    try:
    	f = urlopen(url)
    #handle errors
    except HTTPError, e:
        if e.code == 401:
        	passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
        	if protected_url == '':
        		protected_url = url

        	print 'Install authentication handler for ' + protected_url

        	passman.add_password(None, protected_url, username, password)

        	authhandler = urllib2.HTTPBasicAuthHandler(passman)
        	opener = urllib2.build_opener(authhandler)
        	urllib2.install_opener(opener)

        	f = urlopen(url)
        else:
			print "HTTP Error:", e.code, url
			print e.headers
			return

    except URLError, e:
        print "URL Error:", e.reason, url
        return

    print "-- Downloading " + url + " \n   into " + output

    # Open our local file for writing
    with open(output, write_code) as local_file:
        local_file.write(f.read())
	            

def sample_avi(input, output_dir, sample_ratio, recursive_level):
	# Input is just a file
	if recursive_level == 0:
		try:
			print 'Sample avi file %s into directory %s' % (input, output_dir)

			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

			cap = cv2.VideoCapture(input)

			counter = 0
			while (cap.isOpened()):
				ret, frame = cap.read()

				if not ret:
					return 

				if counter % sample_ratio == 0:
					path = os.path.join(output_dir, '%d.png' % counter)
					resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
					cv2.imwrite(path, resized)
				counter += 1
		except _:
			print 'Exception when process file %s' % (input)
			return
	elif recursive_level > 0:
		if not os.path.isdir(input):
			print '%s is not a directory' % input
			return

		if os.path.exists(output_dir):
			print 'Directory %s exists. Please remove it first.' % output_dir
			return
		else:
			# Create directory
			os.makedirs(output_dir)

		if recursive_level == 1:
			# only check the .avi file
			if os.path.isdir(input):
				for avi_file in glob.glob( os.path.join( input, "*.avi" )):
					sub_output_dir = os.path.join(output_dir, os.path.basename(avi_file)[:-len(".avi")])
					sample_avi(avi_file, sub_output_dir, sample_ratio, 0)
		else:
			for sub_input in glob.glob( os.path.join( input, "*" )):
				sub_output_dir = os.path.join(output_dir, sub_input.split(os.path.sep)[-1])
				sample_avi(sub_input, sub_output_dir, sample_ratio, recursive_level - 1)

def run_densecap( input_dir, output_dir, recursive_level ):
	if 'DENSE_CAP_HOME' not in os.environ:
		print 'Please set DENSE_CAP_HOME=where your put your dense Cap'
		return
	else:
		dense_cap_binary = os.path.join( os.environ['DENSE_CAP_HOME'] , 'run_model.lua' )

	if not os.path.isdir(input_dir):
		print '%s is not a directory' % input_dir
		return

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if recursive_level == 0:
		# Get absolute path, as we have to change dir to $DENSE_CAP_HOME
		input_dir = os.path.abspath(input_dir)
		output_dir = os.path.abspath(output_dir)

		# th run_model.lua -input_dir [Input] -max_images 1000 -output_vis_dir [Output]
		command_line = 'th %s -input_dir %s -max_images 1000 -output_vis_dir %s' \
							% (dense_cap_binary, input_dir, output_dir)
		print command_line
		args = shlex.split(command_line)
		p = subprocess.Popen(args, cwd= os.environ['DENSE_CAP_HOME'])
		p.wait()
		# Remove all images files in the directory (We don't need that)
		return

	if recursive_level > 0:
		for sub_input_dir in glob.glob( os.path.join( input_dir, "*" )):
			sub_output_dir = os.path.join(output_dir, sub_input_dir.split(os.path.sep)[-1])
			run_densecap(sub_input_dir, sub_output_dir, recursive_level - 1)

PROTECTED_URL = 'http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected'
ALL_URLS_URL = PROTECTED_URL + '/downloadLinksAvi.txt'

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='A script to download/process movie snippets')

	username = "5Dvs"
	password = "ma8Aofazah"

	# Download file should be put on the same directory as the process code
	# If couldn't find, download it then
	all_urls = "downloadLinksAvi.txt"

	# For downloading movie snippets from the website
	parser.add_argument('-d', '--download', nargs = 2, metavar = ('MOVIE', 'OUTPUR_DIR'), 
								help = "Download snippets of a movie, such as [American_Beauty]\
	 							into a directory [output_dir]" )
	parser.add_argument('-s', '--sampling', nargs = 4, 
								metavar = ('INPUT', 'OUTPUT_DIR', 'SAMPLE_RATIO', 'RECURSIVE'), 
								help = "Sample from snippets from a movie file/dir [input]\
	 							into a directory [output_dir], [sample_ratio].\
	 							RECURSIVE = 0 means input is just a file, \
	 							RECURSIVE = 1 means process all .avi file in dir [input]\
	 							RECURSIVE = n > 1 process for subdirectories at deep of n-1."  )

	parser.add_argument('-p', '--process', nargs = 4, 
								metavar = ('INPUT', 'OUTPUT_DIR', 'ALGO', 'RECURSIVE'), 
								help = "Process images from [input_dir]\
	 							into a directory [output_dir] using\
	 							algorithm = {'dn': 'darknet', 'dc': 'denseCap'}.\
	 							RECURSIVE = 0 means process all .avi file in a directory\
	 							RECURSIVE = n > 0 process for subdirectories at deep of n."  )

	args = parser.parse_args()

	if args.download and len(args.download) == 2:
		movie_name = args.download[0]
		output_dir = args.download[1]

		if not os.path.exists(all_urls):
			dlfile(ALL_URLS_URL, username, password, all_urls, "w", protected_url = PROTECTED_URL)

		if not os.path.exists(all_urls):
			sys.exit( 'Error: Download downloadLinksAvi.txt failed.' )
		if os.path.exists(output_dir):
			print 'Directory exists. Please remove it first.'
		else:
			# Create directory
			os.makedirs(output_dir)

			with open(all_urls, "r") as all_urls_file:
				for line in all_urls_file:
					url = line.strip()
					if movie_name in url:
						path = os.path.join( output_dir, url.split("?")[0].split(os.path.sep)[-1] )
						dlfile(url, username, password, path, "wb", protected_url = PROTECTED_URL)

	if args.sampling and len(args.sampling) == 4:
		input = args.sampling[0]
		output_dir = args.sampling[1]
		sample_ratio = int(args.sampling[2])
		recursive_level = int(args.sampling[3])

		sample_avi(input, output_dir, sample_ratio, recursive_level)

	if args.process and len(args.process) == 4:
		input_dir = args.process[0]
		output_dir = args.process[1]
		algorithm = args.process[2]
		recursive_level = int(args.process[3])

		if algorithm == 'dn':
			# Run darknet algorithm
			pass

		if algorithm == 'dc':
			# Run denseCap algorithm
			run_densecap(input_dir, output_dir, recursive_level)