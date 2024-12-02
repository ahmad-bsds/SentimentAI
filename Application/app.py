# Flask application
from flask import Flask, render_template, request, redirect, url_for
from Intelligence.analysis import perform_analysis
import requests

app = Flask(__name__)


@app.route('/')
def index():
    data = None
    return render_template('application_interface.html', data=data)


@app.route('/upload', methods=['POST', 'GET'])
def file_upload():
    if request.method == 'POST':
        file = request.files['file']
        file.save('../Data/sentiment_analysis.csv')
        return redirect(url_for('index'))


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        # Retrieve the text entered the textarea
        preferences = request.form.get('preferences')

    # Sentiment
    data = perform_analysis("csv", "sentiment_analysis.csv", preferences=preferences)
    data = data[0].to_dict(orient='list')
    # Recommendations

    recommendations = [
        "Improve customer engagement strategies",
        "Develop targeted marketing campaigns",
        "Focus on product feature refinements",
        "Enhance social media presence"
    ]

    return render_template('application_interface.html', data=data, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
