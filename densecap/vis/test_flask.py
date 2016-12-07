from flask import Flask
from flask import render_template
import os
import glob
import json

app = Flask(__name__)

@app.route("/")
def main():
	return render_template('view_results.html')

if __name__ == "__main__":
	snippets = glob.glob("static/processed/*")

	d = {'snippets' : []}

	for snippet in snippets:
		if os.path.isdir(snippet):
			d['snippets'].append(snippet[len('static/processed/'):])

	with open("static/processed/snippets.json", "w") as file_handler:
		json.dump(d, file_handler)

	app.run(debug = True, threaded = True)
