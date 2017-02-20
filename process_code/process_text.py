import os, sys

import argparse

def change_to_clauseIE( input_line, counter = {'counter' : 0} ):
	description = input_line.split('\t')[-1]

	counter['counter'] += 1
	return '%d\t%s' % (counter['counter'], description)

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='A script to run some commmon processing on text')

	parser.add_argument('-i', '--input', action='store', dest = 'input',
								help = "Input file" )

	parser.add_argument('-o', '--output', action='store', dest = 'output',
								help = "Output file" )

	parser.add_argument('-f', '--function', action='store', dest = 'function',
								help = "A function that turn a string into either None, String or [String]" )

	args = parser.parse_args()

	input = args.input
	output = args.output
	function = args.function

	func = globals()[function]

	output_lines = []

	with open( input, 'r') as fh:
		for line in fh:
			output_line = func( line.strip() )

			if isinstance(output_line, basestring):
				output_lines.append(output_line)

			elif isinstance(output_line, list):
				output_lines += output_line

	with open( output, 'w') as fh:
		for line in output_lines:
			fh.write (line)
			fh.write ('\n')