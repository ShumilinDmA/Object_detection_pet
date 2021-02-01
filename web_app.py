import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run("web_app:app", port=8181, host='0.0.0.0')

