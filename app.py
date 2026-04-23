from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import spacy

app = Flask(__name__)
app.secret_key = "secret123"

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

saved_jobs = []

jobs_data = [
    {
        "job_title": "Data Scientist",
        "skills": "python machine learning pandas data analysis ai",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
        "link": "https://careers.google.com"
    },
    {
        "job_title": "Business Analyst",
        "skills": "excel sql data analysis communication business",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        "link": "https://amazon.jobs"
    },
    {
        "job_title": "Software Engineer",
        "skills": "java c++ python coding development backend",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
        "link": "https://careers.microsoft.com"
    }
]

def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    except:
        pass
    return text.lower()

def extract_skills(text):
    doc = nlp(text)
    skills = []

    skill_keywords = [
        "python", "java", "c++", "machine learning", "deep learning",
        "nlp", "data analysis", "sql", "excel", "pandas",
        "communication", "management", "ai", "backend", "frontend"
    ]

    text_lower = text.lower()

    for skill in skill_keywords:
        if skill in text_lower:
            skills.append(skill)

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            skills.append(token.text.lower())

    return list(set(skills))


@app.route('/')
def home():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    return render_template('index.html', jobs=[], saved=saved_jobs)


@app.route('/recommend', methods=['POST'])
def recommend():

    search = request.form.get('search', '').lower()
    resume_text = ""

    file = request.files.get('pdf')
    if file and file.filename != "":
        resume_text = extract_text_from_pdf(file)

    if not resume_text:
        resume_text = request.form.get('resume', '').lower()

    if not resume_text.strip():
        return render_template('index.html', jobs=[], saved=saved_jobs)

    skills_list = extract_skills(resume_text)
    skills_text = " ".join(skills_list)

    resume_embedding = model.encode(skills_text, convert_to_tensor=True)

    results = []

    for job in jobs_data:

        if search and search not in job['job_title'].lower():
            continue

        job_text = job['job_title'] + " " + job['skills']
        job_embedding = model.encode(job_text, convert_to_tensor=True)

        score = util.cos_sim(resume_embedding, job_embedding).item()
        percent = int(score * 100)

        if percent >= 20:
            results.append({
                "job": job,
                "percent": percent,
                "matched": job['skills'].split(),
                "missing": []
            })

    results.sort(key=lambda x: x["percent"], reverse=True)

    return render_template('index.html', jobs=results, saved=saved_jobs)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    msg = (data.get("message") or "").lower()

    if not msg:
        return jsonify({"reply": "Type something..."})

    user_emb = model.encode(msg, convert_to_tensor=True)

    matches = []

    for job in jobs_data:
        job_text = job['job_title'] + " " + job['skills']
        job_emb = model.encode(job_text, convert_to_tensor=True)

        score = util.cos_sim(user_emb, job_emb).item()
        percent = int(score * 100)

        if percent > 15:
            matches.append(f"{job['job_title']} ({percent}%)")

    if not matches:
        return jsonify({"reply": "No jobs found. Try python, sql, ai."})

    return jsonify({"reply": "\n".join(matches)})


if __name__ == '__main__':
    app.run(debug=True)