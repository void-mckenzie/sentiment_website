from flask import Flask, request, render_template, json, jsonify
import os
app = Flask(__name__,template_folder=os.getcwd()+"\\Deploy") 
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model

model = load_model("Saved_models/senti.h5")

with open('Saved_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)    

def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

length=800

@app.route("/process",methods=["POST"])
def process():
     text = request.form['sent']
     #process here
     new_text=10*func(text)
     return """<html>
    <html>
<head>

  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>

<body>
<nav class="navbar navbar-inverse">
  <div class="container-fluid">
    <div class="navbar-header">
      <a class="navbar-brand" href="#">Sentiment Text Analysis</a>
    </div>
    <ul class="nav navbar-nav">
      <li class="active"><a href="front_sent.html">Home</a></li>
      <li><a href="ps.html">Problem Statement</a></li>
      <li><a href="team.html">The Team</a></li>
    </ul>
  </div>
</nav>
<br><br><br><br>
<center><form action="http://localhost:5000/process" method="POST">
<div class="form-group">
    <label for="rev">Enter your reviews here:</label>
    <textarea class="form-control" rows="5" id="rev" name="sent" style="width: 900px; resize:both;overflow:auto"></textarea>
    
  </div>
<input type="submit" class="btn btn-primary"></button>
</form>
<br><br><br><br>
    <label for="sc">Sentiment Score:</label>
    <p id="sc"></p>
</body>
</html>"""+str(new_text)[0:5]+"""</center></body></html>"""

def func(text):
    f=model.predict(encode_text(tokenizer,[text],length))
    print(f)
    return(f[0][0])
app.run(threaded=False)
