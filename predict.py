import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import render_template
import io

class drowsy:

    def predictdrowsyImage(self, image_file):

        self.image_file = image_file

        model = load_model('outputVGG16_NEW_Architecture.h5')

        basepath = os.path.dirname(__file__) # - org

        file_path = os.path.join(basepath, 'uploads', secure_filename(self.image_file.filename))

        self.image_file.save(file_path)

        test_image = image.load_img(file_path, target_size=(224, 224))  # should be same as given in the code for input
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)  # expand dimension - flattening it

        preds = model.predict(test_image)
        print(preds)

        preds = np.argmax(preds, axis=1)  # The numpy. argmax() function returns indices of the max element of the array in a particular axis.
        print(preds)

        if preds == 0:
            prediction = "alert"
            return prediction
        elif preds == 1:
            prediction = "tired"
            return prediction
        else:
            return "no match found!"

