from app import app
from flask import render_template, request, redirect, url_for
from app.utils.utils import get_efficientdet, preprocessing, make_predictions, save_predictions

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ORIGINAL_IMAGE_ROUTE = 'app/static/test_img.png'
IMG_SIZE = 640
DEVICE = "cpu"

effdet_net = get_efficientdet("app/model/efficientdet_d1.pth")
effdet_net.to(DEVICE)
effdet_net.eval()
print("Model loaded!")


@app.route('/')
def empty():
    return redirect('/index')


@app.route('/index')
def index():
    return render_template('index.html', title='Home', something='Object detection project!')


@app.route('/get_image', methods=["POST"])
def get_image():
    file = request.files['img']
    if file.filename == '':
        return redirect('/index')
    if file and (file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS):
        filename = 'app/static/test_img.png'
        file.save(filename)
    return redirect('/inference')


@app.route('/inference')
def inference():
    img_tensor, height, width = preprocessing(img_path=ORIGINAL_IMAGE_ROUTE, img_size=IMG_SIZE)
    predictions = make_predictions(effdet_net, img_tensor, score_threshold=0.45)
    save_predictions(ORIGINAL_IMAGE_ROUTE, predictions, IMG_SIZE)

    SCALE = 2
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
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
