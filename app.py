from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Tải mô hình từ thư mục lưu trữ trên máy tính của bạn
model_path = 'model/final_model.h5'
model = load_model(model_path)

# Đường dẫn lưu trữ ảnh tải lên
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hàm dự đoán ảnh
def predict_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "PNEUMONIA / VIEM PHOI - PREDICT SCORE:" + str(prediction[0])
    else:
        return "NORMAL / BINH THUONG - PREDICT SCORE:" + str(prediction[0])

# Route cho trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if request.method == 'POST':
        # Kiểm tra xem tệp đã được gửi lên chưa
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Dự đoán ảnh
            prediction = predict_image(file_path, model)
            image_path = file_path.replace('\\', '/')
            image_files.insert(0, filename) 
            image_files = image_files[:10]
            return render_template('index.html', prediction=prediction, image_path=image_path, image_files=image_files)
        image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

    return render_template('index.html', prediction=None, image_files=image_files)

if __name__ == '__main__':
    app.run(debug=True)
