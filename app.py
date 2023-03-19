from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
from play_video import predict_model

app = Flask(__name__)
cap = cv2.VideoCapture(0)
models = []
name_model = ["Family", "Feeling", "Person", "Place", "Pronoun", "Question", "Sick", "Time", "Verb", "Greeting", "Weather"]
data_word = [['Baby', 'Dad', 'Mom'], 
             ['Angry', 'Cry', 'Fine', 'Full', 'Hard work', 'Hungry', 'Sad', 'Scare', 'Smile'], 
             ['Adult', 'Kid', 'Man', 'Person', 'Woman'], 
             ['Hospital', 'House'], 
             ['He', 'Me', 'We'], 
             ['How', 'How much', 'What', 'Where', 'Who', 'Why'], 
             ['Fever', 'Snot'], 
             ['After noon', 'Evening', 'Lunch', 'Morning', 'Night', 'Time'], 
             ['Hello', 'Love', 'Sorry'], 
             ['No', 'No problem', 'Yes'], 
             ['Cold', 'Hot', 'Rain', 'Snow', 'Wind']]
for name in name_model:
    read_model = load_model('data_model/{0}.h5'.format(name), compile=False)
    read_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    models.append(read_model)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/PlayVideo/<string:name>")
def play_video(name):
    return render_template("PlayVideo.html", name=name)

@app.route("/video_feed/<string:name>")
def video_feed(name):
    for num in range(len(name_model)):
        if name == name_model[num]:
            actions = data_word[num]
            model = models[num]
    return Response(predict_model(cap, actions, model), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dataDeveloper")
def dataDeveloper():
    return render_template("dataDeveloper.html")

if __name__=="__main__":
    app.run(debug=True)
