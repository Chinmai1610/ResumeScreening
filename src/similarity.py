from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_vector, jd_vector):
    score = cosine_similarity(resume_vector, jd_vector)
    return round(score[0][0] * 100, 2)
