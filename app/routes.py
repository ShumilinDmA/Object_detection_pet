from app import app
from app.utils.utils import get_efficientdet, preprocessing, make_predictions, save_predictions
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, File, UploadFile, status
import shutil

AUTHOR = 'Dmitrii Shumilin'
GITHUB = 'https://github.com/ShumilinDmA'
PROJECT_REPO = 'https://github.com/ShumilinDmA/Object_detection_pet'
E_mail = 'ShumilinDmAl@gmail.com'
NN_TYPE = 'ssd'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ORIGINAL_IMAGE_ROUTE = 'app/static/test_img.png'
IMG_SIZE = 640
DEVICE = "cpu"
SCALE = 2

templates = Jinja2Templates(directory="app/templates")

effdet_net = get_efficientdet("app/model/efficientdet_d1.pth")
effdet_net.to(DEVICE)
effdet_net.eval()
print("Model loaded!")


@app.get('/', response_class=HTMLResponse)
async def empty():
    """"
    Redirect from default page to index.html
    """
    return RedirectResponse('/index')


@app.get('/index', response_class=HTMLResponse)
async def index(request: Request):
    """
    Render index.html with form for image downloading
    :param request:
    :return:
    """
    return templates.TemplateResponse('/index.html', {'request': request,
                                                      'title': 'Home'})


@app.post('/get_image', response_class=HTMLResponse)
async def get_image(img: UploadFile = File(...)):
    """
    Receive image, check name, extensions, save it as file and redirect to inference.html
    :param img: image file from user
    :return:
    """
    if img.filename == '':
        return RedirectResponse('/index', status_code=status.HTTP_303_SEE_OTHER)
    if img and (img.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS):
        filename = 'app/static/test_img.png'
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
    else:
        return RedirectResponse('/index', status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse('/inference', status_code=status.HTTP_303_SEE_OTHER)


@app.get('/inference', response_class=HTMLResponse)
async def inference(request: Request):
    """
    Performing object detection over user image and render inference.html with predicted results
    :param request:
    :return:
    """
    img_tensor, height, width = preprocessing(img_path=ORIGINAL_IMAGE_ROUTE, img_size=IMG_SIZE)
    predictions = await make_predictions(effdet_net, img_tensor, score_threshold=0.45)
    save_predictions(ORIGINAL_IMAGE_ROUTE, predictions, IMG_SIZE)

    return templates.TemplateResponse('/inference.html',
                                      {'request': request,
                                       'title': 'inference',
                                       'width': width/SCALE,
                                       'height': height/SCALE})


@app.get('/info', response_class=HTMLResponse)
async def info(request: Request):
    """"
    Render some info page about project, author and contacts
    """
    return templates.TemplateResponse('/info.html', {'author': AUTHOR,
                                                     'request': request,
                                                     'title': 'Info',
                                                     'git': GITHUB,
                                                     'repo': PROJECT_REPO,
                                                     'email': E_mail,
                                                     'nn': NN_TYPE})
