from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from retrieval.search_engine import search_hybrid
index_dir = project_root / 'data' / 'index'
stopwords_path = project_root / 'config' / 'stopwords.txt'
queries = [
 'khách sạn view biển đẹp ở phú quốc',
 'resort có hồ bơi và ăn sáng ngon ở đà nẵng',
 'homestay giá rẻ gần chợ đêm đà lạt'
]
for q in queries:
    results, qu = search_hybrid(q, index_dir=index_dir, stopwords_path=stopwords_path, top_k=3)
    print('\nQUERY:', q)
    if not results:
        print('No results or error')
        continue
    for r in results:
        print('-', r.get('hotel_name') or r.get('source_hotel_id'), '|', r.get('location'), '| score', round(r.get('hybrid_score',0),3))
