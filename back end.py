from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        model = load_model("Pneumonia")
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        img = load_img(file, False, target_size=(500, 500))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        return render_template("index1.html", result=str(pred))
    else:
        return render_template("index1.html", result="waiting")


if __name__ == "__main__":
    app.run()
