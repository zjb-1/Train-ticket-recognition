import os
import base64
import cv2 as cv
from flask import Flask, request
from detect_cv_app import me_main

app = Flask(__name__)


def savefile(filename, filepath, data):
    savepath = os.path.join(filepath, filename + '.txt')
    with open(savepath, 'w', encoding='utf-8') as f:
        f.write(data)
        
        
@app.route('/h', methods=['GET', 'POST'])
def hello_world():
    img_path = "/home/zjb/remote/EASTfp_noangle_newaugres/receive.jpg"

    img = request.form['image']
    imgdata = base64.b64decode(img)

    imgfile = open(img_path, 'wb')
    imgfile.write(imgdata)
    imgfile.close()

    recognize = me_main(img_path)
    print(recognize)
    return recognize


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)