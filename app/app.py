# Flask application
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    # render the index templates
    return render_template('application_interface.html')


if __name__ == '__main__':
    app.run(debug=True)
