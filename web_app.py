from app import app

PORT = 7000

if __name__ == "__main__":
    app.run(debug=True, port=PORT, host='localhost')
