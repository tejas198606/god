import os
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__,template_folder='template')

dir_path = os.path.dirname(os.path.realpath(__file__))
## UPLOAD_FOLDER = "uploads"                                ## Folder name
STATIC_FOLDER = "static"                                 ## Folder name


# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "model1.h5")  
## static folder inside model folder and save model 


IMAGE_SIZE = 128                                           ## Image Size can be changed 

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)                          ## channel 3 becaue of R,G,B
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0                                                             # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    
    
    prob = cnn_model.predict(preprocessed_imgage) 
    label = "Corona" if prob[0][0] >= 0.5 else "Normal"                           ## need to rename the features names 
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]

    return label, classified_prob


# home page
@app.route("/")
def home():
    return render_template("home2222.html")                    ## HOME.HTML IS THE TEMPLATE NAME // it can be anything home11.html


@app.route("/classify", methods=["POST", "GET"])           ## classsify.HTML IS THE TEMPLATE NAME // it can be anything home11.html 
def upload_file():

    if request.method == "GET":
        return render_template("home2222.html")  ## HOME.HTML IS THE TEMPLATE NAME // it can be anything home11.html

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(cnn_model, upload_image_path)

        prob = round((prob * 100), 2)

    return render_template(
        "classify2222.html", image_file_name=file.filename, label=label, prob=prob) 
                                          ## classify.html IS THE TEMPLATE NAME // it can be anything classify11.html                    

@app.route("/classify/<filename>")      
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
