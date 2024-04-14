from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from modelLSTM import play_video_lstm
from modelCNN import play_video_cnn
from tensorflow.keras.models import load_model
import torch
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv'}

model_cnn = torch.hub.load('ultralytics/yolov5', 'custom', path='data_model/CNN/Model_CNN_4.pt')
model_lstm = load_model('data_model/LSTM/Model_LSTM_1.h5')

actions = np.array(['Hello', 'Hungry', 'Sick', 'Sorry', 'Thank you', 'What', 'When', 'Where', 'Who', 'Why'])

class VideoForm(FlaskForm):
    video = FileField('Video', validators=[
        FileRequired(),
        FileAllowed(app.config['ALLOWED_EXTENSIONS'], 'Only video files are allowed.')
    ])

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/cnn', methods=['GET', 'POST'])
def indexcnn():
    form = VideoForm()
    process_time = ""
    avg_process = ""
    class_name = ""
    if form.validate_on_submit():
        video = form.video.data
        filename = secure_filename(video.filename)
        video.save(f'static/{app.config["UPLOAD_FOLDER"]}/{filename}')
        process_time, avg_process, class_name = play_video_cnn(filename, model_cnn, actions)
    return render_template('indexcnn.html', form=form, process_time=process_time, avg_process=avg_process, class_name=class_name)

@app.route("/lstm", methods=['GET', 'POST'])
def indexlstm():
    form = VideoForm()
    process_time = ""
    avg_process = ""
    class_name = ""
    if form.validate_on_submit():
        video = form.video.data
        filename = secure_filename(video.filename)
        video.save(f'static/{app.config["UPLOAD_FOLDER"]}/{filename}')
        process_time, avg_process, class_name = play_video_lstm(filename, model_lstm, actions)
    return render_template('indexlstm.html', form=form, process_time=process_time, avg_process=avg_process, class_name=class_name)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dataDeveloper")
def dataDeveloper():
    return render_template("dataDeveloper.html")

@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")

if __name__=="__main__":
    app.run(debug=True)
