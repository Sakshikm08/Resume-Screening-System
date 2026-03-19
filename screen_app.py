from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import nltk
import re
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ── Skill taxonomy ────────────────────────────────────────────────────────────
SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
        "rust", "swift", "kotlin", "php", "scala", "r", "matlab", "perl", "bash"
    ],
    "Web & Frontend": [
        "html", "css", "react", "angular", "vue", "nextjs", "nodejs", "express",
        "django", "flask", "fastapi", "spring", "tailwind", "bootstrap", "graphql",
        "rest", "api", "restful", "webpack", "sass"
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "neural network", "tensorflow",
        "pytorch", "keras", "scikit", "pandas", "numpy", "matplotlib", "seaborn",
        "data analysis", "data science", "computer vision", "regression", "classification",
        "clustering", "reinforcement learning", "bert", "transformers", "llm"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "cassandra", "dynamodb", "firebase", "nosql"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github",
        "gitlab", "ci/cd", "terraform", "ansible", "linux", "unix", "devops",
        "microservices", "serverless", "cloudformation"
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "collaboration", "problem solving",
        "critical thinking", "time management", "adaptability", "creativity",
        "presentation", "mentoring", "agile", "scrum", "project management"
    ],
    "Tools & Platforms": [
        "jira", "confluence", "slack", "figma", "photoshop", "excel", "powerpoint",
        "tableau", "power bi", "jupyter", "vscode", "intellij", "postman"
    ]
}

# Flatten skill list for quick lookup
ALL_SKILLS = {}
for category, skills in SKILL_CATEGORIES.items():
    for skill in skills:
        ALL_SKILLS[skill] = category

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# ── Education & experience patterns ───────────────────────────────────────────
EDUCATION_KEYWORDS = {
    "phd": 5, "doctorate": 5,
    "master": 4, "msc": 4, "mba": 4, "me": 3,
    "bachelor": 3, "bsc": 3, "be": 2, "btech": 3, "bca": 2,
    "diploma": 1, "certification": 1, "certificate": 1
}

EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
    r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:work|industry|professional)',
    r'experience\s+of\s+(\d+)\+?\s*years?',
]

SENIORITY_KEYWORDS = {
    "intern": 0, "junior": 1, "associate": 1,
    "mid": 2, "senior": 3, "lead": 4,
    "principal": 4, "staff": 4, "manager": 4,
    "director": 5, "vp": 5, "head": 5, "chief": 5
}

# ── Text utilities ─────────────────────────────────────────────────────────────
def extract_text(file_bytes, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    return file_bytes.decode('utf-8', errors='ignore')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#\/]', ' ', text)
    return text


def extract_skills(text):
    """Extract matched skills and their categories from text."""
    matched = {}
    text_lower = text.lower()
    for skill, category in ALL_SKILLS.items():
        # Use word-boundary-aware matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            matched[skill] = category
    return matched


def extract_experience_years(text):
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
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


# ── Matching engine ────────────────────────────────────────────────────────────
def compute_match(resume_skills, jd_skills, resume_text, jd_text):
    resume_set = set(resume_skills.keys())
    jd_set     = set(jd_skills.keys())

    matched_skills  = resume_set & jd_set
    missing_skills  = jd_set - resume_set
    bonus_skills    = resume_set - jd_set   # extra skills candidate has

    # Skill match score (weighted: 70% of total)
    skill_score = (len(matched_skills) / len(jd_set) * 100) if jd_set else 0

    # Experience match (15%)
    resume_exp = extract_experience_years(resume_text)
    jd_exp     = extract_experience_years(jd_text)
    if jd_exp == 0:
        exp_score = 80  # no requirement specified → neutral
    elif resume_exp >= jd_exp:
        exp_score = 100
    else:
        exp_score = max(0, (resume_exp / jd_exp) * 100)

    # Education match (15%)
    edu_levels = {
        "Not specified": 0, "Certification": 1, "Certificate": 1, "Diploma": 1,
        "Bca": 2, "Be": 2, "Bsc": 3, "Bachelor": 3, "Btech": 3,
        "Me": 3, "Msc": 4, "Mba": 4, "Master": 4,
        "Doctorate": 5, "Phd": 5
    }
    resume_edu = extract_education_level(resume_text)
    jd_edu     = extract_education_level(jd_text)
    r_level    = edu_levels.get(resume_edu, 0)
    j_level    = edu_levels.get(jd_edu, 0)
    edu_score  = 100 if (j_level == 0 or r_level >= j_level) else max(0, (r_level / j_level) * 100)

    # Weighted total
    total_score = round(skill_score * 0.70 + exp_score * 0.15 + edu_score * 0.15, 1)

    # Recommendation
    if total_score >= 75:
        recommendation = "Strong Match"
        rec_color = "green"
        rec_note  = "This candidate closely matches the job requirements. Recommended for interview."
    elif total_score >= 50:
        recommendation = "Partial Match"
        rec_color = "amber"
        rec_note  = "Candidate meets some requirements. Consider for interview with skill gap awareness."
    else:
        recommendation = "Low Match"
        rec_color = "red"
        rec_note  = "Significant gaps between candidate profile and job requirements."

    # Group matched/missing by category
    def group_by_category(skill_set, source_dict):
        grouped = {}
        for skill in skill_set:
            cat = source_dict.get(skill, "Other")
            grouped.setdefault(cat, []).append(skill)
        return grouped

    return {
        "total_score":       total_score,
        "skill_score":       round(skill_score, 1),
        "exp_score":         round(exp_score, 1),
        "edu_score":         round(edu_score, 1),
        "recommendation":    recommendation,
        "rec_color":         rec_color,
        "rec_note":          rec_note,
        "matched_skills":    sorted(matched_skills),
        "missing_skills":    sorted(missing_skills),
        "bonus_skills":      sorted(list(bonus_skills)[:15]),   # cap at 15
        "matched_by_cat":    group_by_category(matched_skills, {**resume_skills, **jd_skills}),
        "missing_by_cat":    group_by_category(missing_skills, jd_skills),
        "resume_experience": resume_exp,
        "jd_experience":     jd_exp,
        "resume_education":  resume_edu,
        "jd_education":      jd_edu,
        "resume_seniority":  extract_seniority(resume_text),
        "jd_seniority":      extract_seniority(jd_text),
        "total_jd_skills":   len(jd_set),
        "total_matched":     len(matched_skills),
        "total_missing":     len(missing_skills),
        "total_bonus":       len(bonus_skills),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'screen_index.html')


@app.route('/screen', methods=['POST'])
def screen():
    if 'resume' not in request.files or 'jd' not in request.files:
        return jsonify({'error': 'Both resume and job description files are required.'}), 400

    resume_file = request.files['resume']
    jd_file     = request.files['jd']
    candidate   = request.form.get('candidate_name', 'Candidate').strip() or 'Candidate'
    job_title   = request.form.get('job_title', 'Position').strip() or 'Position'

    try:
        resume_bytes = resume_file.read()
        jd_bytes     = jd_file.read()

        resume_text  = extract_text(resume_bytes, resume_file.filename)
        jd_text      = extract_text(jd_bytes, jd_file.filename)

        resume_skills = extract_skills(resume_text)
        jd_skills     = extract_skills(jd_text)

        if not jd_skills:
            return jsonify({'error': 'Could not extract any skills from the job description. Please check the file content.'}), 422

        result = compute_match(resume_skills, jd_skills, resume_text, jd_text)
        result['candidate_name'] = candidate
        result['job_title']      = job_title

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("AI Resume Screening System running at http://localhost:5001")
    app.run(debug=True, port=5001)