from app import app
from flask import render_template, request, redirect, url_for

from PIL import Image


@app.route('/')
def empty():
    return redirect('/index')


@app.route('/index')
def index():
    return render_template('index.html', title='Home', something='Object detection project!')


@app.route('/get_image', methods=["POST"])
def get_image():
    file = request.files['img']
    print(request.files)
    print(file.filename)
    if file.filename == '':
        return redirect('/index')
    image = Image.open(file, 'r')
    image.save('app/static/test_img.png', 'PNG')
    return redirect('/inference')


@app.route('/inference')
def inference():
    ORIGINAL_IMAGE_ROUTE = 'app/static/test_img.png'
    PREDICTED_IMAGE_ROUTE = 'app/static/pred_img.png'
    SCALE = 3
    original_image = Image.open(ORIGINAL_IMAGE_ROUTE)
    predicted_image = Image.open(PREDICTED_IMAGE_ROUTE)
    width, height = original_image.size
    original_img_url = url_for('static', filename='test_img.png')
    predicted_img_url = url_for('static', filename='pred_img.png')
    return render_template('inference.html',
                           original_img_url=original_img_url,
                           predicted_img_url=predicted_img_url,
                           width=width/SCALE,
                           height=height/SCALE)


@app.route('/info')
def info():
    AUTHOR = 'Dmitrii Shumilin'
    GITHUB = 'https://github.com/ShumilinDmA'
    PROJECT_REPO = 'https://github.com/ShumilinDmA/Object_detection_pet'
    E_mail = 'ShumilinDmAl@gmail.com'
    NN_TYPE = 'ssd'
    NN_IMG_URL = url_for('static', filename='network.png')
    return render_template("info.html",
                           author=AUTHOR,
                           git=GITHUB,
                           repo=PROJECT_REPO,
                           email=E_mail,
                           nn=NN_TYPE,
                           nn_img=NN_IMG_URL)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
