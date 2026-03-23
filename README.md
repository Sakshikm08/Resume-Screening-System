AI Resume Screening System

An AI-powered web app that analyzes resumes against job descriptions using NLP and weighted scoring to help recruiters shortlist candidates quickly.

Features

Single & multi-resume screening
Match score (Skills 70%, Experience 15%, Education 15%)
Skill gap analysis (matched, missing, bonus)
AI-generated summary & interview questions
Learning resources for missing skills
Leaderboard ranking
Export report & email sharing

Tech Stack
Python, Flask, NLTK, PyMuPDF, HTML, CSS, JavaScript, Chart.js

Setup

Install dependencies:
pip install flask nltk PyMuPDF
Run the app:
python screen_app.py
Open:
http://localhost:5001

Usage
Upload resume(s) and a job description to get analysis, ranking, and hiring recommendations.

Scoring
Total Score = 70% Skills + 15% Experience + 15% Education
