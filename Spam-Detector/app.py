from flask import Flask, render_template, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        count_vect = joblib.load('./count_vect.pkl')
        NB_spam_model = joblib.load('./NB_spam_model.pkl')
        message = request.form['message']
        data = [message]
        data = count_vect.transform(data).toarray()
        my_prediction = NB_spam_model.predict(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
