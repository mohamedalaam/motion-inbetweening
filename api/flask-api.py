from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask import send_from_directory
import os
from src.utils import get_project_path
ROOT_DIR=get_project_path()
from model.inference import  InferModel
app = Flask(__name__)

model=InferModel()


@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def upload_file_1():
    if request.method == 'POST':
        f = request.files['file']
        print(secure_filename(f.filename))
        path_to_save=os.path.join(ROOT_DIR,'api','uploaded_files', secure_filename(f.filename))
        f.save(path_to_save)
        data=model.load_BVH_file('uploaded_files')
        print(data)
        return 'file uploaded successfully'

@app.route('/downloads')
def download_file():
    return send_from_directory('uploaded_files',
                               'run2_subject4.bvh', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
