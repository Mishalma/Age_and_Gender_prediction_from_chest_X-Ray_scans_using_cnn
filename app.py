from flask import Flask, render_template, request
import numpy as np
import os
from model import image_pre,predict


app = Flask(__name__)


UPLOAD_FOLDER = 'C:/Users/moham/lungsbased_age_gender_prediction/static'
ALLOWED_EXTENSIONS = set(['png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
        file1.save(path)
        data = image_pre(path)
        age,gen = predict(data)
        if gen == 1:
            result = 'Predicted age is '+str(age)+' years and the person is a Male'
        else:
            result = 'Predicted age is '+str(age)+' years and the person is a Female'
    return render_template('index.html',result=result) 





if __name__ == "__main__":
    app.run(debug=True)