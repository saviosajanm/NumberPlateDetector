from flask import Flask, render_template, request, redirect, url_for, flash, abort, session, jsonify, Blueprint
import json
import os.path
from werkzeug.utils import secure_filename
from model import PyImageSearchANPR
import glob
import base64
import io
import os
from PIL import Image

app = Flask(__name__)

@app.route("/")
def home():
    for i in  glob.glob("imgs/*.png"):
        os.remove(i)
    for i in glob.glob("imgs1/*.png"):
        print(PyImageSearchANPR.process(i), "+++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    for i in glob.glob("imgs2/*.png"):
        print(PyImageSearchANPR.process(i), "))))))))))))))))))))))))))))))))))))))))))))))))))))))")
    return render_template("home.html")

@app.route("/numberplate", methods = ['GET', 'POST'])
def numplate():
    if request.method == 'POST':
      f = request.files['file']
      f.save("imgs/" + f.filename)
      if glob.glob("imgs/*.png") == []:
            if glob.glob("imgs/*.jpg") != []:
                im1 = Image.open(glob.glob("imgs/*.jpg")[0])
            elif glob.glob("imgs/*.jpeg") != []:
                im1 = Image.open(glob.glob("imgs/*.jpeg")[0])
            im1.save("imgs/" + f.filename[: f.filename.index(".")] + ".png")
            os.remove("imgs/" + f.filename)
      #print(type(f))
      data = io.BytesIO()
      im = Image.open("imgs/" + f.filename[: f.filename.index(".")] + ".png")
      im.save(data, "PNG")
      encoded_img_data = base64.b64encode(data.getvalue())
      print(glob.glob("imgs/*.png")[0], "=================================")
      text = PyImageSearchANPR.process(glob.glob("imgs/*.png")[0])
      print(text[1], "-----------------------------------------------------------------------------------")
      return render_template("numplate.html", np=text[1], img_data=encoded_img_data.decode('utf-8'))
