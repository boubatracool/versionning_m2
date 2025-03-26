import os
from flask import Flask, render_template
import upload_dataset as upload_dataset

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'MASTER2IA_VERSIONING2')


@app.route("/")
def hello_world():
    return render_template('hello.html')

@app.route('/upload', methods=['POST'])
def upload():
    UPLOAD_FOLDER = 'data'
    ALLOWED_EXTENSIONS = {'csv'}
    return upload_dataset.upload_file(UPLOAD_FOLDER,ALLOWED_EXTENSIONS)


if __name__ == '__main__':
    app.run(debug=True)
