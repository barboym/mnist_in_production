from flask import Flask
from flask import render_template,redirect,url_for,request
app = Flask(__name__)



@app.route("/")
def home(res=None):
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def model():
    # request.
    return redirect(url_for('home',res=7))


if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)
