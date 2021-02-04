from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title='Object Detection project',
              version='2.0',
              description='Pet project in computer vision and deployment')

app.mount("/static", StaticFiles(directory="app/static"), name="static")

from app import routes
