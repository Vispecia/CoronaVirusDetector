from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=='POST':
        mydict = request.form
        breath = int(mydict['Breath'])
        fever = float(mydict['Fever'])
        nose = int(mydict['RunnyNose'])
        age = int(mydict['Age'])
        pain = int(mydict['BodyPain'])
        inputFeature = [fever,age,breath,nose,pain]
        inf = clf.predict_proba([inputFeature])[0][1]
        inf = int(inf*100)
        return render_template('result.html', inf=inf)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True) 