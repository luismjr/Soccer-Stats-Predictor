from flask import Flask, render_template, send_from_directory
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),  # optional for CSS/JS/images
)

@app.get("/")
def home():
    # This will render templates/index.html
    return render_template("index.html")

@app.get("/data/<path:filename>")
def data_files(filename):
    # This serves files from your data/ folder
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)
