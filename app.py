from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)


model = load_model('./fake.h5')

voc_size = 15000
sent_length = 50

dict = {0: 'Fake', 1: 'Real'}

# routes
@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("home.html")



@app.route("/submit", methods = ['GET', 'POST'])
def news():
	if request.method == 'POST':
		data = request.form['news']

		onehotrepr = [one_hot(data, voc_size)]
		embedded_docs = pad_sequences(onehotrepr, padding='pre', maxlen = sent_length)

		news = np.argmax(model.predict(embedded_docs), axis=1)[0]

	return render_template("home.html", news = dict[news])


if __name__ =='__main__':
	app.run(debug = True)