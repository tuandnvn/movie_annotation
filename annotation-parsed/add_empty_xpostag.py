import sys

if __name__ == '__main__':
	input = sys.argv[1]
	output = sys.argv[2]

	results = []
	with open(input, 'r') as input_handler:
		for line in input_handler:
			tokens = line.strip().split('\t')
			if len(tokens) == 9:
				new_tokens = tokens[:4] + ['_'] + tokens[4:]
				new_line = '\t'.join(new_tokens)
			else:
				new_line = ''
			results.append(new_line)

	with open(output, 'w') as output_handler:
		for line in results:
			output_handler.write(line)
			output_handler.write('\n')