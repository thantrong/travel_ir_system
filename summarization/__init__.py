"""
Review Summarization
PHẦN 8: REVIEW SUMMARIZATION
Extractive Summarization using TF-IDF / TextRank.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def split_sentences(text: str) -> list[str]:
    """Tách đoạn văn thành các câu dựa trên dấu chấm, hỏi, than ôi."""
    if not text:
        return []
    # Thay thế nhiều khoảng trắng thành 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    # Tách câu đơn giản bằng regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def summarize_reviews_tfidf(reviews: list[str], top_n_sentences: int = 2) -> str:
    """
    Extractive Summarization bằng cách tính độ quan trọng của câu sử dụng TF-IDF.
    Ý tưởng bao gồm:
    1. Chuyển toàn bộ reviews thành tập hợp các câu.
    2. Vector hóa các câu bằng TF-IDF.
    3. Graph-based ranking (TextRank) hoặc tính điểm trung bình cosine similarity với các câu khác.
    Ở đây dùng phương pháp TextRank đơn giản hóa: 
    Điểm của 1 câu = Tổng độ tương đồng (Cosine Similarity) với tất cả các câu khác.
    """
    if not reviews:
        return ""

    all_sentences = []
    for r in reviews:
        all_sentences.extend(split_sentences(r))

    if not all_sentences:
        return ""

    # Nếu số lượng câu ít hơn cấu hình, trả về toàn bộ
    if len(all_sentences) <= top_n_sentences:
        return " ".join(all_sentences)

    # 1. Xây dựng ma trận TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
    except ValueError:
        # Trong trường hợp vocabulary empty (toàn stop words hoặc không hợp lệ)
        return " ".join(all_sentences[:top_n_sentences])

    # 2. Xây dựng ma trận Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 3. Tính điểm cho mỗi câu (tổng độ tương đồng với các câu khác)
    # Đây là 1 dạng đơn giản của PageRank / TextRank
    scores = np.zeros(len(all_sentences))
    for i in range(len(all_sentences)):
        scores[i] = similarity_matrix[i].sum() - similarity_matrix[i, i] # Trừ đi điểm với chính nó

    # 4. Sắp xếp và lấy top câu
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(all_sentences)), reverse=True)
    
    # Sắp xếp lại top N câu theo thứ tự xuất hiện ban đầu để văn bản mạch lạc hơn
    top_indices = []
    for i in range(min(top_n_sentences, len(ranked_sentences))):
        top_s = ranked_sentences[i][1]
        top_indices.append(all_sentences.index(top_s))
        
    top_indices.sort()
    
    summary = " ".join([all_sentences[i] for i in top_indices])
    return summary

