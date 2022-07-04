from flask import Flask, render_template,request
from tensorflow.keras import models
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

label =  {0:"benign",1:"malignant"}
model = models.load_model('./model/Model.h5')
app = Flask(__name__)

@app.route('/',methods=['GET'])
def HelloWorld():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path="./static/" + imagefile.filename
    imagefile.save(image_path)
    img_ = image.load_img(image_path, target_size=(224, 224))
    imag = image.img_to_array(img_)
    imag = np.expand_dims(imag, axis=0)
    pred = model.predict(imag)
    pred = np.argmax(pred,axis=1)
    return render_template('index.html',prediction=label[pred[0]],imagePath=image_path)


if __name__ == '__main__':
    app.run(port=3000,debug=True)

