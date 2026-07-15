from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from src.models.base import BaseRecommender

class MarkovRecommender(BaseRecommender):
    def __init__(self, config=None):
        self.config = config or {}
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.candidate_pois = set()
        self.model_name = 'markov'

    def fit(self, data, valid_data=None):
        """
        训练一阶马尔可夫模型，使用统一的编码后索引。
        """
        print(f"开始训练 {self.model_name} 模型...")
        train_df = data.train
        self.candidate_pois = set(data.candidate_poi_ids)
        
        # 💡 核心修改：按编码后的索引排序和分组
        train_df = train_df.sort_values(by=['user_idx', 'utc_time'])
        
        # 统计转移频次
        transfer_counts = defaultdict(lambda: defaultdict(int))
        poi_out_counts = defaultdict(int)
        
        # 💡 核心修改：使用 'user_idx' 和 'poi_idx'
        user_groups = train_df.groupby('user_idx')['poi_idx'].apply(list).to_dict()
        
        for user, pois in tqdm(user_groups.items(), desc="统计状态转移频次"):
            if len(pois) < 2:
                continue
            for i in range(len(pois) - 1):
                current_poi = pois[i]
                next_poi = pois[i+1]
                
                transfer_counts[current_poi][next_poi] += 1
                poi_out_counts[current_poi] += 1
                
        # 计算转移概率
        for current_poi, next_pois in tqdm(transfer_counts.items(), desc="计算转移概率矩阵"):
            total_out = poi_out_counts[current_poi]
            for next_poi, count in next_pois.items():
                self.transition_matrix[current_poi][next_poi] = count / total_out
                
        print("Markov 训练完成！")

    def recommend(self, data, top_k=10):
        """
        预测：基于滚动历史的最后一个 POI，推荐转移概率最高的前 K 个候选 POI。
        """
        print(f"开始使用 {self.model_name} 生成标准推荐...")
        results = []
        
        # 兜底策略：使用全局热门 POI
        popular_pois = [poi for poi in data.candidate_poi_ids[:top_k]]
        
        samples = list(data.iter_next_poi_samples("test", max_history=50))
        
        for sample in tqdm(samples, desc="生成评估预测表"):
            event_id = sample.event_id
            history = sample.history
            
            recommended = []
            
            # 如果历史记录不为空，取最后一个 POI 作为当前状态
            if history and len(history) > 0:
                last_poi = history[-1]
                if last_poi in self.transition_matrix:
                    # 获取该 POI 转移概率最高的前 K 个 POI
                    sorted_trans = sorted(
                        self.transition_matrix[last_poi].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    recommended = [poi for poi, prob in sorted_trans if poi in self.candidate_pois]
            
            # 补齐与去重，确保恰好输出 10 个合法的候选 POI
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
            
            # 写入结果
            for rank, poi_idx in enumerate(final_recommendations, start=1):
                results.append({
                    'event_id': event_id,
                    'rank': rank,
                    'poi_idx': poi_idx
                })
                
        return pd.DataFrame(results)