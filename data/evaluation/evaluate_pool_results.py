"""
Script đánh giá hệ thống retrieval từ pool_results_labeled.csv
- Đánh giá từng mô hình: BM25, Vector, Hybrid
- Tính P@5, P@10, MAP@10, NDCG@10 cho từng bucket và tổng kết
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

# Đường dẫn file
LABELED_FILE = Path(__file__).parent / "pool_results_labeled.csv"
OUTPUT_FILE = Path(__file__).parent / "evaluation_results.txt"


def compute_precision_at_k(rels, k):
    """Tính Precision@k"""
    if k == 0:
        return 0
    return sum(rels[:k]) / k


def compute_ap_at_k(rels, k=10):
    """Tính Average Precision@k"""
    hits = 0
    sum_precisions = 0
    for i in range(min(k, len(rels))):
        if rels[i] == 1:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / k if k > 0 else 0


def compute_ndcg_at_k(rels, k=10):
    """Tính NDCG@k"""
    dcg = sum(rels[i] / math.log2(i + 2) for i in range(min(k, len(rels))))
    
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(ideal_rels[i] / math.log2(i + 2) for i in range(min(k, len(ideal_rels))))
    
    return dcg / idcg if idcg > 0 else 0


def evaluate():
    """Đánh giá hệ thống từ file labeled"""
    if not LABELED_FILE.exists():
        print(f"Error: {LABELED_FILE} not found")
        return
    
    # Đọc file và group theo query
    queries_data = defaultdict(lambda: {"bm25": [], "vector": [], "hybrid": [], "bucket": ""})
    
    with open(LABELED_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id", "")
            bucket_id = row.get("bucket_id", "")
            relevant = int(row.get("relevant", 0))
            
            bm25_rank = int(row.get("bm25_rank", 999)) if row.get("bm25_rank", "").strip() else 999
            vector_rank = int(row.get("vector_rank", 999)) if row.get("vector_rank", "").strip() else 999
            hybrid_rank = int(row.get("hybrid_rank", 999)) if row.get("hybrid_rank", "").strip() else 999
            
            queries_data[query_id]["bucket"] = bucket_id
            queries_data[query_id]["bm25"].append((bm25_rank, relevant))
            queries_data[query_id]["vector"].append((vector_rank, relevant))
            queries_data[query_id]["hybrid"].append((hybrid_rank, relevant))
    
    # Sort by rank
    for qid in queries_data:
        for model in ["bm25", "vector", "hybrid"]:
            queries_data[qid][model].sort(key=lambda x: x[0])
    
    # Group by bucket
    bucket_queries = defaultdict(list)
    for qid, data in queries_data.items():
        bucket_queries[data["bucket"]].append(qid)
    
    # Evaluate each model per bucket
    models = ["bm25", "vector", "hybrid"]
    metrics_names = ["P@5", "P@10", "MAP@10", "NDCG@10"]
    
    all_bucket_results = {}
    all_model_results = {m: {"P@5": [], "P@10": [], "MAP@10": [], "NDCG@10": []} for m in models}
    
    print("=" * 80)
    print("EVALUATION RESULTS BY BUCKET AND MODEL")
    print("=" * 80)
    
    for bucket_id in sorted(bucket_queries.keys()):
        qids = bucket_queries[bucket_id]
        bucket_results = {}
        
        for model in models:
            p5_scores = []
            p10_scores = []
            map10_scores = []
            ndcg10_scores = []
            
            for qid in qids:
                rels = [rel for rank, rel in queries_data[qid][model]]
                
                p5 = compute_precision_at_k(rels, 5)
                p10 = compute_precision_at_k(rels, 10)
                map10 = compute_ap_at_k(rels, 10)
                ndcg10 = compute_ndcg_at_k(rels, 10)
                
                p5_scores.append(p5)
                p10_scores.append(p10)
                map10_scores.append(map10)
                ndcg10_scores.append(ndcg10)
                
                all_model_results[model]["P@5"].append(p5)
                all_model_results[model]["P@10"].append(p10)
                all_model_results[model]["MAP@10"].append(map10)
                all_model_results[model]["NDCG@10"].append(ndcg10)
            
            bucket_results[model] = {
                "P@5": sum(p5_scores) / len(p5_scores),
                "P@10": sum(p10_scores) / len(p10_scores),
                "MAP@10": sum(map10_scores) / len(map10_scores),
                "NDCG@10": sum(ndcg10_scores) / len(ndcg10_scores),
            }
        
        all_bucket_results[bucket_id] = bucket_results
        
        print(f"\nBucket: {bucket_id} ({len(qids)} queries)")
        print("-" * 70)
        print(f"{'Model':<12} {'P@5':>8} {'P@10':>8} {'MAP@10':>8} {'NDCG@10':>8}")
        print("-" * 70)
        for model in models:
            m = bucket_results[model]
            print(f"{model:<12} {m['P@5']:>8.4f} {m['P@10']:>8.4f} {m['MAP@10']:>8.4f} {m['NDCG@10']:>8.4f}")
    
    # Overall results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS (ALL BUCKETS)")
    print("=" * 80)
    print(f"{'Model':<12} {'P@5':>8} {'P@10':>8} {'MAP@10':>8} {'NDCG@10':>8}")
    print("-" * 70)
    
    overall_results = {}
    for model in models:
        overall_results[model] = {
            "P@5": sum(all_model_results[model]["P@5"]) / len(all_model_results[model]["P@5"]),
            "P@10": sum(all_model_results[model]["P@10"]) / len(all_model_results[model]["P@10"]),
            "MAP@10": sum(all_model_results[model]["MAP@10"]) / len(all_model_results[model]["MAP@10"]),
            "NDCG@10": sum(all_model_results[model]["NDCG@10"]) / len(all_model_results[model]["NDCG@10"]),
        }
        m = overall_results[model]
        print(f"{model:<12} {m['P@5']:>8.4f} {m['P@10']:>8.4f} {m['MAP@10']:>8.4f} {m['NDCG@10']:>8.4f}")
    
    # Best model highlight
    best_model = max(overall_results, key=lambda m: overall_results[m]["NDCG@10"])
    print(f"\n=> Best model: {best_model} (NDCG@10: {overall_results[best_model]['NDCG@10']:.4f})")
    
    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS BY BUCKET AND MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        for bucket_id in sorted(bucket_queries.keys()):
            f.write(f"Bucket: {bucket_id} ({len(bucket_queries[bucket_id])} queries)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<12} {'P@5':>8} {'P@10':>8} {'MAP@10':>8} {'NDCG@10':>8}\n")
            f.write("-" * 60 + "\n")
            for model in models:
                m = all_bucket_results[bucket_id][model]
                f.write(f"{model:<12} {m['P@5']:>8.4f} {m['P@10']:>8.4f} {m['MAP@10']:>8.4f} {m['NDCG@10']:>8.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL RESULTS (ALL BUCKETS)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<12} {'P@5':>8} {'P@10':>8} {'MAP@10':>8} {'NDCG@10':>8}\n")
        f.write("-" * 60 + "\n")
        for model in models:
            m = overall_results[model]
            f.write(f"{model:<12} {m['P@5']:>8.4f} {m['P@10']:>8.4f} {m['MAP@10']:>8.4f} {m['NDCG@10']:>8.4f}\n")
        f.write(f"\n=> Best model: {best_model} (NDCG@10: {overall_results[best_model]['NDCG@10']:.4f})\n")
    
    print(f"\nSaved detailed results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()