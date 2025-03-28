import os
from flask import Flask, request, render_template, jsonify
import upload_dataset as upload_dataset

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'MASTER2IA_VERSIONING2')


@app.route("/")
def index():
    return render_template('./templates/index.html')


@app.route('/upload', methods=['POST'])
def upload():
    UPLOAD_FOLDER = 'data'
    ALLOWED_EXTENSIONS = {'csv'}

    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')  # Retrieve multiple files

    uploaded_files = []
    for file in files:
        if file.filename == '':
            continue  # Skip empty filenames
        
        if file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
        else:
            return jsonify({'error': f'File {file.filename} has an invalid extension'}), 400

    return jsonify({'message': 'Files uploaded successfully', 'files': uploaded_files}), 200


if __name__ == '__main__':
    app.run(debug=True)
