# src/registry.py
from src.models.itemcf import ItemCF  # 👈 确保导入了你的模型
from src.models.markov import MarkovRecommender

MODEL_REGISTRY = {

    "itemcf": ItemCF,                  # 👈 确保这一行对应你的 ItemCF 类
    "markov": MarkovRecommender,       # 👈 确保这一行对应你的 MarkovRecommender 类
    # ... 其他模型的映射 ...
}