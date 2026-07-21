import os
import argparse
import time
import pandas as pd
# 1. 替换为团队最新更新的配置加载工具
from src.utils.config import load_yaml  
from src.datasets import load_data_bundle
from src.registry import MODEL_REGISTRY
from src.evaluator import evaluate_next_poi  # 2. 替换为团队统一的评价函数


def parse_args():
    parser = argparse.ArgumentParser(description="POI 推荐系统实验运行入口")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="配置文件路径，例如 configs/itemcf.yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. 加载配置（使用统一的 load_yaml）
    print(f"正在加载配置文件: {args.config}")
    config = load_yaml(args.config)
    model_name = config.get("model_name")
    
    # 2. 加载数据
    print("正在加载数据...")
    data = load_data_bundle("configs/data.yaml")
    
    # 3. 从注册表构建模型
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{model_name}' 未在 src/registry.py 中注册！")
    
    print(f"正在构建模型: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config)
    
    # 4. 训练模型
    start_train = time.time()
    # 统一接口传入 data 对象，让模型内部灵活提取 train_data 或 candidate_pois
    model.fit(data)
    train_time = time.time() - start_train
    print(f"模型训练完成，耗时: {train_time:.2f} 秒")
    
    # 5. 生成预测（生成团队要求的预测格式）
    start_infer = time.time()
    print("开始为测试集生成推荐...")
    prediction_frame = model.recommend(data, top_k=10)
    inference_time = time.time() - start_infer
    print(f"预测完成，耗时: {inference_time:.2f} 秒")
    
    # 6. 创建输出目录
    pred_dir = "results/predictions"
    metrics_dir = "results/metrics"
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 7. 保存预测结果 (CSV)
    pred_path = os.path.join(pred_dir, f"{model_name}.csv")
    prediction_frame.to_csv(pred_path, index=False)
    print(f"🎉 预测结果已成功保存至: {pred_path}")
    
    # 8. 使用团队统一指标接口进行评估
    try:
        print("正在进行统一指标评估...")
        metrics = evaluate_next_poi(
            targets=data.test,  # 传入测试集目标
            predictions=prediction_frame,
            candidate_poi_ids=data.candidate_poi_ids,
            unknown_id=data.unknown_id,
            ks=(5, 10),
            mrr_k=10,
        )
        
        # 补充耗时与模型信息
        metrics["model"] = model_name
        metrics["train_time"] = round(train_time, 2)
        metrics["inference_time"] = round(inference_time, 2)
        
        # 保存指标为 JSON
        import json
        metrics_path = os.path.join(metrics_dir, f"{model_name}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"🎉 评估指标已成功保存至: {metrics_path}")
        
    except Exception as e:
        print(f"⚠️ 评估保存失败，错误信息: {e}")


if __name__ == "__main__":
    main()