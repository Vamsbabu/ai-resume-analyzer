# =============================================================
# job_matcher.py - NLP Analysis & ATS Score Calculation
# Uses spaCy + scikit-learn to compare resume vs job description
# =============================================================

import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ─────────────────────────────────────────────
# Load spaCy English NLP model
# ─────────────────────────────────────────────
# We use the small English model for speed and efficiency.
# Make sure to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run this command: python -m spacy download en_core_web_sm"
    )


# ─────────────────────────────────────────────
# Core NLP Functions
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Preprocess text using spaCy NLP:
    - Tokenize text into words
    - Lowercase everything
    - Remove stopwords (common words like 'the', 'is', 'at')
    - Remove punctuation
    - Lemmatize words (e.g., 'running' → 'run')

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, lemmatized text joined as a single string.
    """

    # Process the text with spaCy
    doc = nlp(text.lower())

    # Keep only meaningful tokens: no stopwords, no punctuation, no spaces
    tokens = [
        token.lemma_          # Use the base form of the word
        for token in doc
        if not token.is_stop           # Remove stopwords
        and not token.is_punct         # Remove punctuation
        and not token.is_space         # Remove whitespace tokens
        and len(token.text) > 1        # Remove single characters
    ]

    return " ".join(tokens)


def extract_keywords(text: str, top_n: int = 20) -> list:
    """
    Extract the most important keywords from text using spaCy.
    Uses Part-of-Speech tagging to identify nouns and proper nouns,
    which are typically the most meaningful keywords in resumes/JDs.

    Args:
        text (str): Input text (resume or job description).
        top_n (int): Number of top keywords to return.

    Returns:
        list: Top N keywords sorted by frequency.
    """

    doc = nlp(text.lower())

    # Extract nouns, proper nouns, and adjectives — these carry the most meaning
    keywords = [
        token.lemma_
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ")   # Parts of speech
        and not token.is_stop
        and not token.is_punct
        and len(token.text) > 2
    ]

    # Count keyword frequency and return the most common ones
    keyword_counts = Counter(keywords)
    top_keywords = [word for word, count in keyword_counts.most_common(top_n)]

    return top_keywords


def find_matching_keywords(resume_text: str, jd_text: str) -> dict:
    """
    Compare keywords between resume and job description.

    Returns:
        dict: {
            "matched": [...],      # Keywords found in both
            "missing": [...],      # Keywords in JD but NOT in resume
            "match_count": int,
            "total_jd_keywords": int
        }
    """

    resume_keywords = set(extract_keywords(resume_text, top_n=40))
    jd_keywords = set(extract_keywords(jd_text, top_n=40))

    # Find intersection and difference
    matched = resume_keywords.intersection(jd_keywords)
    missing = jd_keywords.difference(resume_keywords)

    return {
        "matched": sorted(list(matched)),
        "missing": sorted(list(missing)),
        "match_count": len(matched),
        "total_jd_keywords": len(jd_keywords),
    }


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate TF-IDF cosine similarity between two texts.

    TF-IDF (Term Frequency-Inverse Document Frequency) weighs how
    important a word is relative to the document and corpus.
    Cosine similarity then measures the angle between the two
    document vectors — 1.0 = identical, 0.0 = completely different.

    Args:
        text1 (str): Preprocessed resume text.
        text2 (str): Preprocessed job description text.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """

    # TfidfVectorizer converts text into numerical vectors
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # Consider both single words and 2-word phrases
        max_features=5000,      # Limit vocabulary size for efficiency
        sublinear_tf=True,      # Apply log normalization to term frequency
    )

    # Fit and transform both texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Return as a float (similarity is a 1x1 matrix)
    return float(similarity[0][0])


# ─────────────────────────────────────────────
# Main ATS Score Function
# ─────────────────────────────────────────────

def calculate_ats_score(resume_text: str, job_description: str) -> dict:
    """
    Main function: Analyzes resume against job description
    and returns a comprehensive ATS compatibility report.

    Steps:
    1. Preprocess both texts
    2. Calculate TF-IDF cosine similarity
    3. Extract and compare keywords
    4. Compute a weighted ATS score
    5. Generate feedback and recommendations

    Args:
        resume_text (str): Full text extracted from the PDF resume.
        job_description (str): Job description pasted by the user.

    Returns:
        dict: Full ATS analysis report with score and recommendations.
    """

    # ── Step 1: Preprocess both texts ──
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(job_description)

    # ── Step 2: Calculate cosine similarity (50% of score) ──
    similarity_score = calculate_cosine_similarity(processed_resume, processed_jd)

    # ── Step 3: Keyword matching analysis (50% of score) ──
    keyword_data = find_matching_keywords(resume_text, job_description)

    # Keyword match ratio
    if keyword_data["total_jd_keywords"] > 0:
        keyword_ratio = keyword_data["match_count"] / keyword_data["total_jd_keywords"]
    else:
        keyword_ratio = 0.0

    # ── Step 4: Compute weighted final ATS score ──
    # 50% cosine similarity + 50% keyword match ratio
    raw_score = (similarity_score * 0.50) + (keyword_ratio * 0.50)

    # Scale to 0–100 and apply a calibration multiplier for realism
    # Raw TF-IDF scores tend to be low; we scale them to a human-readable range
    ats_score = min(round(raw_score * 180, 1), 100.0)

    # ── Step 5: Generate rating label ──
    rating, rating_color = get_rating(ats_score)

    # ── Step 6: Generate actionable recommendations ──
    recommendations = generate_recommendations(ats_score, keyword_data)

    # ── Step 7: Word count stats ──
    resume_word_count = len(resume_text.split())
    jd_word_count = len(job_description.split())

    return {
        "ats_score": ats_score,
        "rating": rating,
        "rating_color": rating_color,
        "similarity_score": round(similarity_score * 100, 1),
        "keyword_match_score": round(keyword_ratio * 100, 1),
        "matched_keywords": keyword_data["matched"][:15],      # Top 15 for display
        "missing_keywords": keyword_data["missing"][:15],      # Top 15 for display
        "match_count": keyword_data["match_count"],
        "total_jd_keywords": keyword_data["total_jd_keywords"],
        "resume_word_count": resume_word_count,
        "jd_word_count": jd_word_count,
        "recommendations": recommendations,
    }


def get_rating(score: float) -> tuple:
    """
    Convert numeric ATS score into a human-readable rating label and color.

    Args:
        score (float): ATS score 0–100.

    Returns:
        tuple: (rating_label, color_hex)
    """
    if score >= 80:
        return "Excellent Match", "#00C48C"
    elif score >= 65:
        return "Good Match", "#3B82F6"
    elif score >= 50:
        return "Fair Match", "#F59E0B"
    elif score >= 35:
        return "Needs Improvement", "#F97316"
    else:
        return "Poor Match", "#EF4444"


def generate_recommendations(score: float, keyword_data: dict) -> list:
    """
    Generate actionable, personalized recommendations based on the
    ATS score and keyword gap analysis.

    Args:
        score (float): ATS score.
        keyword_data (dict): Keyword matching results.

    Returns:
        list: List of recommendation strings.
    """
    recommendations = []
    missing = keyword_data["missing"][:8]  # Focus on top missing keywords

    # ── Keyword-based recommendations ──
    if missing:
        kw_list = ", ".join(missing[:5])
        recommendations.append(
            f"Add these missing keywords to your resume: {kw_list}."
        )

    # ── Score-based recommendations ──
    if score < 40:
        recommendations.append(
            "Your resume has low alignment with this job. Consider rewriting your summary "
            "and skills sections to better reflect the job requirements."
        )
        recommendations.append(
            "Tailor your work experience bullet points to mirror the language used in the job description."
        )

    elif score < 60:
        recommendations.append(
            "Incorporate more job-specific terminology in your skills section and work experience."
        )
        recommendations.append(
            "Review the job description carefully and ensure your top 3 achievements align with the role's key requirements."
        )

    elif score < 75:
        recommendations.append(
            "Good alignment! Fine-tune your professional summary to echo the job title and core skills mentioned in the posting."
        )
        recommendations.append(
            "Quantify your achievements with numbers (e.g., 'Increased efficiency by 30%') to stand out."
        )

    else:
        recommendations.append(
            "Great match! Ensure your resume is formatted cleanly with standard headings (Experience, Education, Skills) for ATS parsing."
        )
        recommendations.append(
            "Double-check that your contact info, LinkedIn URL, and key certifications are clearly listed."
        )

    # ── Universal best practices ──
    recommendations.append(
        "Avoid tables, graphics, headers/footers, and columns — ATS systems often fail to parse these correctly."
    )
    recommendations.append(
        "Use a standard font (Arial, Calibri, Times New Roman) and save your resume as a text-based PDF."
    )

    return recommendations
