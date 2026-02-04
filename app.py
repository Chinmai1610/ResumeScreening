from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    score = None

    if request.method == "POST":
        resume = request.form["resume"]
        job = request.form["job"]

        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform([resume, job])

        similarity = cosine_similarity(vectors)[0][1]
        score = round(similarity * 100, 2)

    return render_template("index.html", score=score)

if __name__ == "__main__":
    app.run(debug=True)
