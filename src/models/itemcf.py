import math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from src.models.base import BaseRecommender

class ItemCF(BaseRecommender):
    def __init__(self, config=None):
        self.config = config or {}
        self.k = self.config.get('k', 50)
        self.sim_matrix = defaultdict(dict)
        self.user_history = dict()
        self.candidate_pois = set()
        self.model_name = 'itemcf'

    def fit(self, data, valid_data=None):
        """
        训练 ItemCF 模型，使用统一的编码后索引 `user_idx` 和 `poi_idx`。
        """
        print(f"开始训练 {self.model_name} 模型...")
        train_df = data.train
        self.candidate_pois = set(data.candidate_poi_ids)
        
        # 💡 核心修改：将 'user_id' 和 'poi_id' 替换为 'user_idx' 和 'poi_idx'
        user_items = train_df.groupby('user_idx')['poi_idx'].apply(list).to_dict()
        self.user_history = {u: set(items) for u, items in user_items.items()}
        
        item_counts = defaultdict(int)
        co_occur = defaultdict(lambda: defaultdict(int))
        
        # 统计共现
        for user, items in tqdm(self.user_history.items(), desc="构建共现矩阵"):
            for i in items:
                item_counts[i] += 1
                for j in items:
                    if i == j:
                        continue
                    co_occur[i][j] += 1
                    
        # 计算余弦相似度
        for i, related_items in tqdm(co_occur.items(), desc="计算余弦相似度"):
            for j, cuv in related_items.items():
                sim = cuv / math.sqrt(item_counts[i] * item_counts[j])
                self.sim_matrix[i][j] = sim
                
        # 截断 Top-K
        for i in self.sim_matrix:
            sorted_items = sorted(self.sim_matrix[i].items(), key=lambda x: x[1], reverse=True)[:self.k]
            self.sim_matrix[i] = dict(sorted_items)
            
        print("ItemCF 训练完成！")

    def recommend(self, data, top_k=10):
        """
        按照团队统一接口生成预测：
        1. 必须使用滚动历史 samples 迭代器
        2. 输出格式严格限定为：event_id, rank, poi_idx
        3. 推荐结果必须全部在候选集中
        """
        print(f"开始使用 {self.model_name} 生成标准推荐...")
        results = []
        
        # 默认推荐列表（当用户无历史或相似度矩阵无法覆盖时，用训练集最热的 POI 兜底）
        # 保证每个事件都能输出 10 个合法的、互不重复的候选 POI
        popular_pois = [poi for poi in data.candidate_poi_ids[:top_k]]
        
        # 遍历测试集滚动样本 (对应评估目标)
        # iter_next_poi_samples 会产生带 event_id 的测试流
        samples = list(data.iter_next_poi_samples("test", max_history=50))
        
        for sample in tqdm(samples, desc="生成评估预测表"):
            event_id = sample.event_id  # 统一评价需要的事件 ID
            history = sample.history    # 滚动历史列表
            
            scores = defaultdict(float)
            
            # 基于滚动历史进行 ItemCF 推荐得分累加
            for hist_item in history:
                if hist_item in self.sim_matrix:
                    for related_item, sim_score in self.sim_matrix[hist_item].items():
                        # 确保推荐的 POI 属于官方候选集
                        if related_item in self.candidate_pois:
                            scores[related_item] += sim_score
            
            # 排序获取推荐
            sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            recommended = [item for item, score in sorted_res]
            
            # 兜底策略：如果推荐数不足 top_k，用最热 POI 补齐并去重，确保恰好 10 个
            final_recommendations = []
            seen = set()
            for poi in recommended:
                if poi not in seen and poi in self.candidate_pois:
                    final_recommendations.append(poi)
                    seen.add(poi)
                if len(final_recommendations) == top_k:
                    break
                    
            if len(final_recommendations) < top_k:
                for poi in popular_pois:
                    if poi not in seen:
                        final_recommendations.append(poi)
                        seen.add(poi)
                    if len(final_recommendations) == top_k:
                        break
            
            # 确保推荐结果始终在合法范围内
            for rank, poi_idx in enumerate(final_recommendations, start=1):
                # 如果模型不小心推荐了 PAD(0) 或 UNK(1)，直接排除
                if poi_idx < 2:
                    continue 
                results.append({
                    'event_id': event_id,
                    'rank': rank,
                    'poi_idx': poi_idx
                })
                
        return pd.DataFrame(results)