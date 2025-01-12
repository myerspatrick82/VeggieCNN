from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import torchvision.transforms.v2 as transforms
from PIL import Image
import torch
from veggienet import VeggieNet

# declare some cool stuff
app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'
app.config['UPLOAD_FOLDER'] = 'static/files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
classes = ('Bean', 'Ladies Finger', 'Onion', 'Pointed Gourd', 'Potato', 'Eggplant') # How classes are setup in output of Linear layer

# load model in
model = VeggieNet()
state_dict = torch.load('veggieCNN.pt', weights_only=True)
model.load_state_dict(state_dict=state_dict)
model.eval()

# class
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

# image preprocess
def transform_img(image):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                                    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])])
    return transform(image)

@app.route('/home', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # grab file
        # print(file.filename) # Then save the file
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename)) # saving file
        prediction = predict(filename)
        return prediction
        
    return render_template('index.html', form=form)

def predict(image_name):
    path = r'./static/files/' + image_name
    img = Image.open(path)
    img = transform_img(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    softmax_output = torch.nn.functional.softmax(outputs)
    val, predicted = torch.max(softmax_output, 1)
    # print(outputs)
    prediction = classes[predicted.item()]
    return f'{val.item()*100:.2f}% sure of a(n) {prediction}'



if __name__=='__main__':
    app.run(debug=True)