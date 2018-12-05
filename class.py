import pandas as pd
from flask import render_template
from flask import Flask, request, redirect, url_for
from sklearn.tree import DecisionTreeClassifier

#Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def pageRender():

    #Clasificador
    df = pd.read_csv('dataSet.csv')

    features = df.drop('Label', axis=1).values
    label = df['Label'].values

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(features, label)
    #Clasificador End

    if request.method == 'POST':
        color = request.form['color']
        distance = request.form['distance']

        X = [[color,distance]]
        r = dt.predict(X)

        return render_template("resultado.html", resultado=r[0])

    return render_template("home.html")

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=8000)
#Flask End