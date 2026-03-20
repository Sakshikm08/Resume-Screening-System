from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

app = Flask(__name__)

LEADERBOARD = []

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python","java","javascript","typescript","c++","c#","ruby","go",
        "rust","swift","kotlin","php","scala","r","matlab","perl","bash"
    ],
    "Web & Frontend": [
        "html","css","react","angular","vue","nextjs","nodejs","express",
        "django","flask","fastapi","spring","tailwind","bootstrap","graphql",
        "rest","api","restful","webpack","sass"
    ],
    "Data & ML": [
        "machine learning","deep learning","nlp","neural network","tensorflow",
        "pytorch","keras","scikit","pandas","numpy","matplotlib","seaborn",
        "data analysis","data science","computer vision","regression","classification",
        "clustering","reinforcement learning","bert","transformers","llm"
    ],
    "Databases": [
        "sql","mysql","postgresql","mongodb","redis","elasticsearch",
        "sqlite","oracle","cassandra","dynamodb","firebase","nosql"
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","docker","kubernetes","jenkins","git","github",
        "gitlab","ci/cd","terraform","ansible","linux","unix","devops",
        "microservices","serverless","cloudformation"
    ],
    "Soft Skills": [
        "leadership","communication","teamwork","collaboration","problem solving",
        "critical thinking","time management","adaptability","creativity",
        "presentation","mentoring","agile","scrum","project management"
    ],
    "Tools & Platforms": [
        "jira","confluence","slack","figma","photoshop","excel","powerpoint",
        "tableau","power bi","jupyter","vscode","intellij","postman"
    ]
}

ALL_SKILLS = {}
for category, skills in SKILL_CATEGORIES.items():
    for skill in skills:
        ALL_SKILLS[skill] = category

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))

LEARNING_RESOURCES = {
    "python":           {"platform":"Python.org",   "url":"https://docs.python.org/3/tutorial/"},
    "java":             {"platform":"Codecademy",   "url":"https://www.codecademy.com/learn/learn-java"},
    "javascript":       {"platform":"MDN",          "url":"https://developer.mozilla.org/en-US/docs/Learn/JavaScript"},
    "typescript":       {"platform":"TypeScript",   "url":"https://www.typescriptlang.org/docs/"},
    "react":            {"platform":"React Docs",   "url":"https://react.dev/learn"},
    "nodejs":           {"platform":"Node.js",      "url":"https://nodejs.org/en/learn"},
    "django":           {"platform":"Django",       "url":"https://docs.djangoproject.com/en/stable/intro/tutorial01/"},
    "flask":            {"platform":"Flask Docs",   "url":"https://flask.palletsprojects.com/en/stable/tutorial/"},
    "fastapi":          {"platform":"FastAPI",      "url":"https://fastapi.tiangolo.com/tutorial/"},
    "machine learning": {"platform":"Coursera",    "url":"https://www.coursera.org/learn/machine-learning"},
    "deep learning":    {"platform":"fast.ai",      "url":"https://course.fast.ai/"},
    "nlp":              {"platform":"Hugging Face", "url":"https://huggingface.co/learn/nlp-course/"},
    "tensorflow":       {"platform":"TensorFlow",   "url":"https://www.tensorflow.org/tutorials"},
    "pytorch":          {"platform":"PyTorch",      "url":"https://pytorch.org/tutorials/"},
    "scikit":           {"platform":"scikit-learn", "url":"https://scikit-learn.org/stable/getting_started.html"},
    "pandas":           {"platform":"Pandas",       "url":"https://pandas.pydata.org/getting_started.html"},
    "numpy":            {"platform":"NumPy",        "url":"https://numpy.org/learn/"},
    "sql":              {"platform":"SQLZoo",        "url":"https://sqlzoo.net/"},
    "postgresql":       {"platform":"PostgreSQL",   "url":"https://www.postgresql.org/docs/current/tutorial.html"},
    "mongodb":          {"platform":"MongoDB Univ", "url":"https://learn.mongodb.com/"},
    "aws":              {"platform":"AWS Training", "url":"https://aws.amazon.com/training/"},
    "azure":            {"platform":"MS Learn",     "url":"https://learn.microsoft.com/en-us/training/azure/"},
    "gcp":              {"platform":"Google Cloud", "url":"https://cloud.google.com/learn/training"},
    "docker":           {"platform":"Docker Docs",  "url":"https://docs.docker.com/get-started/"},
    "kubernetes":       {"platform":"Kubernetes",   "url":"https://kubernetes.io/docs/tutorials/"},
    "git":              {"platform":"Git SCM",      "url":"https://git-scm.com/book/en/v2"},
    "linux":            {"platform":"Linux Journey","url":"https://linuxjourney.com/"},
    "agile":            {"platform":"Atlassian",    "url":"https://www.atlassian.com/agile"},
    "scrum":            {"platform":"Scrum.org",    "url":"https://www.scrum.org/resources/what-scrum-module"},
    "tableau":          {"platform":"Tableau",      "url":"https://www.tableau.com/learn/training"},
    "power bi":         {"platform":"Microsoft",    "url":"https://learn.microsoft.com/en-us/power-bi/"},
    "jenkins":          {"platform":"Jenkins",      "url":"https://www.jenkins.io/doc/tutorials/"},
    "terraform":        {"platform":"HashiCorp",    "url":"https://developer.hashicorp.com/terraform/tutorials"},
}

INTERVIEW_QUESTIONS = {
    "python":           ["What are Python decorators and how do you use them?","Explain the difference between list and tuple.","What is the GIL in Python?"],
    "java":             ["What is the difference between JDK, JRE, and JVM?","Explain Java garbage collection.","What are Java generics?"],
    "javascript":       ["What is event delegation in JavaScript?","Explain the difference between == and ===.","What is a closure?"],
    "react":            ["What is the virtual DOM in React?","Explain the difference between props and state.","What are React hooks?"],
    "machine learning": ["What is the bias-variance tradeoff?","Explain supervised vs unsupervised learning.","How do you handle imbalanced datasets?"],
    "deep learning":    ["What is backpropagation?","Explain the vanishing gradient problem.","Difference between CNN and RNN?"],
    "nlp":              ["What is tokenization?","Explain TF-IDF.","Difference between stemming and lemmatization?"],
    "tensorflow":       ["What is a tensor?","Explain eager vs graph execution.","How do you save and load models?"],
    "pytorch":          ["What is autograd in PyTorch?","Explain .detach() vs .no_grad().","How do you build a custom dataset?"],
    "scikit":           ["What is cross-validation?","How does Random Forest work?","What is feature scaling?"],
    "pandas":           ["Difference between loc and iloc?","How do you handle missing values?","Difference between merge and join?"],
    "sql":              ["Difference between INNER JOIN and LEFT JOIN?","What are SQL indexes?","Explain WHERE vs HAVING."],
    "postgresql":       ["What are PostgreSQL schemas?","How does MVCC work?","Difference between SERIAL and IDENTITY?"],
    "mongodb":          ["Embedding vs referencing in MongoDB?","How do MongoDB indexes work?","What is an aggregation pipeline?"],
    "aws":              ["Difference between EC2 and Lambda?","How does S3 versioning work?","What is IAM?"],
    "docker":           ["Difference between Docker image and container?","What is a Dockerfile?","How do Docker volumes work?"],
    "kubernetes":       ["Difference between Pod and Deployment?","How does Kubernetes handle scaling?","What is a Kubernetes service?"],
    "git":              ["Difference between git merge and rebase?","How do you resolve merge conflicts?","What is git stash?"],
    "linux":            ["Difference between process and thread in Linux?","How do you check running processes?","What are Linux file permissions?"],
    "agile":            ["What are Agile core principles?","What is a sprint retrospective?","How do you handle scope creep?"],
    "flask":            ["How does Flask handle routing?","What is the application context?","How do you handle file uploads?"],
    "fastapi":          ["What makes FastAPI faster than Flask?","How does FastAPI handle data validation?","Path params vs query params?"],
    "django":           ["What is Django ORM?","Explain the MTV architecture.","What is middleware in Django?"],
    "docker":           ["Image vs container?","What is a Dockerfile?","How do volumes work?"],
    "git":              ["Merge vs rebase?","How to resolve conflicts?","What is git stash?"],
}

DEFAULT_QUESTIONS = [
    "Describe a challenging project you have worked on.",
    "How do you stay up to date with new technologies?",
    "Tell me about a time you had to learn a new skill quickly.",
]

EDUCATION_KEYWORDS = {
    "phd":5,"doctorate":5,"master":4,"msc":4,"mba":4,"me":3,
    "bachelor":3,"bsc":3,"be":2,"btech":3,"bca":2,
    "diploma":1,"certification":1,"certificate":1
}

EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
    r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:work|industry|professional)',
    r'experience\s+of\s+(\d+)\+?\s*years?',
]

SENIORITY_KEYWORDS = {
    "intern":0,"junior":1,"associate":1,"mid":2,"senior":3,
    "lead":4,"principal":4,"staff":4,"manager":4,
    "director":5,"vp":5,"head":5,"chief":5
}

EDU_LEVELS = {
    "Not specified":0,"Certification":1,"Certificate":1,"Diploma":1,
    "Bca":2,"Be":2,"Bsc":3,"Bachelor":3,"Btech":3,
    "Me":3,"Msc":4,"Mba":4,"Master":4,"Doctorate":5,"Phd":5
}


def extract_text(file_bytes, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    return file_bytes.decode('utf-8', errors='ignore')


def extract_skills(text):
    matched = {}
    text_lower = text.lower()
    for skill, category in ALL_SKILLS.items():
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            matched[skill] = category
    return matched


def extract_experience_years(text):
    for pattern in EXPERIENCE_PATTERNS:
        m = re.search(pattern, text.lower())
        if m:
            return int(m.group(1))
    return 0


def extract_education_level(text):
    text_lower = text.lower()
    best = (0, "Not specified")
    for keyword, level in EDUCATION_KEYWORDS.items():
        if keyword in text_lower and level > best[0]:
            best = (level, keyword.capitalize())
    return best[1]


def extract_seniority(text):
    text_lower = text.lower()
    best = (-1, "Not specified")
    for keyword, level in SENIORITY_KEYWORDS.items():
        if re.search(r'\b' + keyword + r'\b', text_lower) and level > best[0]:
            best = (level, keyword.capitalize())
    return best[1]


def generate_summary(candidate, job_title, score, rec, matched, missing, exp, edu, seniority):
    matched_str = ', '.join(matched[:5]) if matched else 'general skills'
    missing_str = ', '.join(missing[:3]) if missing else 'none critical'
    exp_str     = f"{exp} year(s) of experience" if exp else "experience not explicitly stated"
    level_str   = seniority if seniority != "Not specified" else "mid-level"

    if score >= 75:
        opening  = f"{candidate} is a strong candidate for the {job_title} role, achieving an overall match score of {score}%."
        strength = f"The candidate demonstrates solid expertise in {matched_str}, aligning well with the core requirements."
        gap_note = f"Minor gaps exist in {missing_str}, which could be addressed through brief onboarding." if missing else "No significant skill gaps were identified."
        closing  = "Highly recommended for interview."
    elif score >= 50:
        opening  = f"{candidate} is a partial match for the {job_title} role with a match score of {score}%."
        strength = f"Key strengths include {matched_str}, covering a portion of the role's requirements."
        gap_note = f"Notable gaps in {missing_str} should be discussed during the interview."
        closing  = "Consider for interview with awareness of skill gaps."
    else:
        opening  = f"{candidate} shows limited alignment with the {job_title} role, scoring {score}%."
        strength = f"The candidate has some relevant skills including {matched_str}."
        gap_note = f"Significant gaps in {missing_str} may require substantial upskilling."
        closing  = "May not be suitable for this role at this time."

    profile = f"Profile: {level_str.capitalize()} professional with {exp_str}, holding a {edu} degree."
    return f"{opening} {profile} {strength} {gap_note} {closing}"


def generate_interview_questions(missing_skills, matched_skills):
    questions = []
    for skill in missing_skills[:4]:
        qs = INTERVIEW_QUESTIONS.get(skill)
        if qs:
            questions.append({"skill": skill, "type": "gap", "question": qs[0]})
    for skill in matched_skills[:3]:
        qs = INTERVIEW_QUESTIONS.get(skill)
        if qs:
            questions.append({"skill": skill, "type": "verify", "question": qs[0]})
    for q in DEFAULT_QUESTIONS:
        if len(questions) >= 8:
            break
        questions.append({"skill": "General", "type": "general", "question": q})
    return questions[:8]


def get_learning_resources(missing_skills):
    return [
        {"skill": s, "platform": LEARNING_RESOURCES[s]["platform"], "url": LEARNING_RESOURCES[s]["url"]}
        for s in missing_skills if s in LEARNING_RESOURCES
    ]


def compute_match(resume_skills, jd_skills, resume_text, jd_text, candidate="Candidate", job_title="Position"):
    resume_set = set(resume_skills.keys())
    jd_set     = set(jd_skills.keys())

    matched_skills = resume_set & jd_set
    missing_skills = jd_set - resume_set
    bonus_skills   = resume_set - jd_set

    skill_score = (len(matched_skills) / len(jd_set) * 100) if jd_set else 0
    resume_exp  = extract_experience_years(resume_text)
    jd_exp      = extract_experience_years(jd_text)
    exp_score   = 80 if jd_exp == 0 else (100 if resume_exp >= jd_exp else max(0, resume_exp / jd_exp * 100))
    resume_edu  = extract_education_level(resume_text)
    jd_edu      = extract_education_level(jd_text)
    r_level     = EDU_LEVELS.get(resume_edu, 0)
    j_level     = EDU_LEVELS.get(jd_edu, 0)
    edu_score   = 100 if (j_level == 0 or r_level >= j_level) else max(0, r_level / j_level * 100)
    total_score = round(skill_score * 0.70 + exp_score * 0.15 + edu_score * 0.15, 1)

    if total_score >= 75:
        recommendation, rec_color = "Strong Match", "green"
        rec_note = "This candidate closely matches the job requirements. Recommended for interview."
    elif total_score >= 50:
        recommendation, rec_color = "Partial Match", "amber"
        rec_note = "Candidate meets some requirements. Consider for interview with skill gap awareness."
    else:
        recommendation, rec_color = "Low Match", "red"
        rec_note = "Significant gaps between candidate profile and job requirements."

    def group_by_cat(skill_set, src):
        g = {}
        for s in skill_set:
            c = src.get(s, "Other")
            g.setdefault(c, []).append(s)
        return g

    matched_sorted = sorted(matched_skills)
    missing_sorted = sorted(missing_skills)
    seniority      = extract_seniority(resume_text)

    return {
        "candidate_name":      candidate,
        "job_title":           job_title,
        "total_score":         total_score,
        "skill_score":         round(skill_score, 1),
        "exp_score":           round(exp_score, 1),
        "edu_score":           round(edu_score, 1),
        "recommendation":      recommendation,
        "rec_color":           rec_color,
        "rec_note":            rec_note,
        "matched_skills":      matched_sorted,
        "missing_skills":      missing_sorted,
        "bonus_skills":        sorted(list(bonus_skills)[:15]),
        "matched_by_cat":      group_by_cat(matched_skills, {**resume_skills, **jd_skills}),
        "missing_by_cat":      group_by_cat(missing_skills, jd_skills),
        "resume_experience":   resume_exp,
        "jd_experience":       jd_exp,
        "resume_education":    resume_edu,
        "jd_education":        jd_edu,
        "resume_seniority":    seniority,
        "jd_seniority":        extract_seniority(jd_text),
        "total_jd_skills":     len(jd_set),
        "total_matched":       len(matched_skills),
        "total_missing":       len(missing_skills),
        "total_bonus":         len(bonus_skills),
        "summary":             generate_summary(candidate, job_title, total_score, recommendation,
                                                matched_sorted, missing_sorted, resume_exp, resume_edu, seniority),
        "interview_questions": generate_interview_questions(missing_sorted, matched_sorted),
        "learning_resources":  get_learning_resources(missing_sorted),
    }


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'screen_index.html')


@app.route('/screen', methods=['POST'])
def screen():
    if 'resume' not in request.files or 'jd' not in request.files:
        return jsonify({'error': 'Both resume and job description files are required.'}), 400
    resume_file = request.files['resume']
    jd_file     = request.files['jd']
    candidate   = request.form.get('candidate_name', 'Candidate').strip() or 'Candidate'
    job_title   = request.form.get('job_title', 'Position').strip() or 'Position'
    try:
        resume_text   = extract_text(resume_file.read(), resume_file.filename)
        jd_text       = extract_text(jd_file.read(),     jd_file.filename)
        resume_skills = extract_skills(resume_text)
        jd_skills     = extract_skills(jd_text)
        if not jd_skills:
            return jsonify({'error': 'Could not extract skills from the job description.'}), 422
        result = compute_match(resume_skills, jd_skills, resume_text, jd_text, candidate, job_title)
        LEADERBOARD.append({"candidate_name": candidate, "job_title": job_title,
                             "total_score": result['total_score'], "recommendation": result['recommendation'],
                             "rec_color": result['rec_color'], "matched": result['total_matched'],
                             "missing": result['total_missing']})
        LEADERBOARD.sort(key=lambda x: x['total_score'], reverse=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/screen-multi', methods=['POST'])
def screen_multi():
    if 'jd' not in request.files:
        return jsonify({'error': 'Job description file is required.'}), 400
    jd_file   = request.files['jd']
    job_title = request.form.get('job_title', 'Position').strip() or 'Position'
    jd_text   = extract_text(jd_file.read(), jd_file.filename)
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return jsonify({'error': 'Could not extract skills from the job description.'}), 422

    resumes = request.files.getlist('resumes')
    names   = request.form.getlist('candidate_names')
    results = []

    for i, resume_file in enumerate(resumes):
        candidate = names[i].strip() if i < len(names) and names[i].strip() else f"Candidate {i+1}"
        try:
            resume_text   = extract_text(resume_file.read(), resume_file.filename)
            resume_skills = extract_skills(resume_text)
            r = compute_match(resume_skills, jd_skills, resume_text, jd_text, candidate, job_title)
            results.append(r)
            LEADERBOARD.append({"candidate_name": candidate, "job_title": job_title,
                                 "total_score": r['total_score'], "recommendation": r['recommendation'],
                                 "rec_color": r['rec_color'], "matched": r['total_matched'],
                                 "missing": r['total_missing']})
        except Exception as e:
            results.append({"candidate_name": candidate, "error": str(e), "total_score": 0})

    LEADERBOARD.sort(key=lambda x: x['total_score'], reverse=True)
    results.sort(key=lambda x: x.get('total_score', 0), reverse=True)
    return jsonify({"results": results, "job_title": job_title})


@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    return jsonify(LEADERBOARD[:20])


@app.route('/leaderboard/clear', methods=['POST'])
def clear_leaderboard():
    LEADERBOARD.clear()
    return jsonify({"status": "cleared"})


@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.get_json()
    to_email     = data.get('to_email', '').strip()
    sender_email = data.get('sender_email', '').strip()
    sender_pass  = data.get('sender_password', '').strip()
    report       = data.get('report', '')
    subject      = data.get('subject', 'AI Resume Screening Report')
    if not all([to_email, sender_email, sender_pass, report]):
        return jsonify({'error': 'Missing required fields.'}), 400
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = sender_email
        msg['To']      = to_email
        msg.attach(MIMEText(report, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, to_email, msg.as_string())
        return jsonify({'status': 'Email sent successfully!'})
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Use an App Password.'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("AI Resume Screening System running at http://localhost:5001")
    app.run(debug=True, port=5001)