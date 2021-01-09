from flask import Flask

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# python -m flask run -h localhost -p 81

from app import routes
