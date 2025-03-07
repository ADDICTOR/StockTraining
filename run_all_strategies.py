#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行所有选股策略并比较结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# 导入各个策略
from value_strategy_basic import ValueBasicStrategy
from growth_strategy import GrowthStrategy
from quality_strategy import QualityStrategy
from momentum_strategy import MomentumStrategy
from multi_factor_strategy import MultiFactorStrategy

def run_all_strategies():
    """运行所有策略并比较结果"""
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行各个策略
    print("=" * 50)
    print("开始运行所有选股策略...")
    print("=" * 50)
    
    # 运行双低双高价值选股策略
    print("\n1. 运行双低双高价值选股策略...")
    value_strategy = ValueBasicStrategy(
        pe_threshold=15,
        pb_threshold=1.5,
        roe_threshold=10,
        dividend_threshold=2,
        top_n=20
    )
    value_stocks = value_strategy.run()
    
    # 运行成长型选股策略
    print("\n2. 运行成长型选股策略...")
    growth_strategy = GrowthStrategy(
        profit_growth_threshold=20,
        revenue_growth_threshold=15,
        roe_growth_threshold=5,
        eps_growth_threshold=15,
        top_n=20
    )
    growth_stocks = growth_strategy.run()
    
    # 运行质量型选股策略
    print("\n3. 运行质量型选股策略...")
    quality_strategy = QualityStrategy(
        roe_threshold=15,
        gross_margin_threshold=30,
        debt_to_asset_threshold=50,
        cash_flow_quality_threshold=1.0,
        top_n=20
    )
    quality_stocks = quality_strategy.run()
    
    # 运行动量型选股策略
    print("\n4. 运行动量型选股策略...")
    momentum_strategy = MomentumStrategy(
        momentum_3m_weight=0.5,
        momentum_6m_weight=0.3,
        momentum_12m_weight=0.2,
        top_n=20
    )
    momentum_stocks = momentum_strategy.run()
    
    # 运行多因子选股策略
    print("\n5. 运行多因子选股策略...")
    multi_factor_strategy = MultiFactorStrategy(
        value_weight=0.25,
        growth_weight=0.25,
        quality_weight=0.25,
        momentum_weight=0.25,
        top_n=20
    )
    multi_factor_stocks = multi_factor_strategy.run()
    
    # 计算各策略之间的重叠股票
    print("\n" + "=" * 50)
    print("分析各策略之间的重叠股票...")
    
    # 提取各策略选出的股票代码
    value_codes = set(value_stocks['code'])
    growth_codes = set(growth_stocks['code'])
    quality_codes = set(quality_stocks['code'])
    momentum_codes = set(momentum_stocks['code'])
    multi_factor_codes = set(multi_factor_stocks['code'])
    
    # 计算重叠情况
    value_growth_overlap = value_codes.intersection(growth_codes)
    value_quality_overlap = value_codes.intersection(quality_codes)
    value_momentum_overlap = value_codes.intersection(momentum_codes)
    growth_quality_overlap = growth_codes.intersection(quality_codes)
    growth_momentum_overlap = growth_codes.intersection(momentum_codes)
    quality_momentum_overlap = quality_codes.intersection(momentum_codes)
    
    # 计算与多因子策略的重叠
    value_multi_overlap = value_codes.intersection(multi_factor_codes)
    growth_multi_overlap = growth_codes.intersection(multi_factor_codes)
    quality_multi_overlap = quality_codes.intersection(multi_factor_codes)
    momentum_multi_overlap = momentum_codes.intersection(multi_factor_codes)
    
    # 输出重叠情况
    print(f"\n价值策略与成长策略重叠股票数: {len(value_growth_overlap)}")
    print(f"价值策略与质量策略重叠股票数: {len(value_quality_overlap)}")
    print(f"价值策略与动量策略重叠股票数: {len(value_momentum_overlap)}")
    print(f"成长策略与质量策略重叠股票数: {len(growth_quality_overlap)}")
    print(f"成长策略与动量策略重叠股票数: {len(growth_momentum_overlap)}")
    print(f"质量策略与动量策略重叠股票数: {len(quality_momentum_overlap)}")
    
    print(f"\n价值策略与多因子策略重叠股票数: {len(value_multi_overlap)}")
    print(f"成长策略与多因子策略重叠股票数: {len(growth_multi_overlap)}")
    print(f"质量策略与多因子策略重叠股票数: {len(quality_multi_overlap)}")
    print(f"动量策略与多因子策略重叠股票数: {len(momentum_multi_overlap)}")
    
    # 找出被多个策略共同选中的股票
    all_selected_codes = list(value_codes) + list(growth_codes) + list(quality_codes) + list(momentum_codes)
    code_counts = {}
    for code in all_selected_codes:
        if code in code_counts:
            code_counts[code] += 1
        else:
            code_counts[code] = 1
    
    # 找出被至少3个策略选中的股票
    highly_selected = {code: count for code, count in code_counts.items() if count >= 3}
    
    if highly_selected:
        print("\n被至少3个策略共同选中的股票:")
        # 获取这些股票的名称
        all_stocks = pd.concat([value_stocks, growth_stocks, quality_stocks, momentum_stocks])
        all_stocks = all_stocks.drop_duplicates(subset=['code'])
        
        for code, count in highly_selected.items():
            stock_name = all_stocks.loc[all_stocks['code'] == code, 'name'].values[0] if len(all_stocks.loc[all_stocks['code'] == code]) > 0 else "未知"
            print(f"代码: {code}, 名称: {stock_name}, 被选中次数: {count}")
    else:
        print("\n没有被至少3个策略共同选中的股票")
    
    # 记录结束时间并计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print(f"所有策略运行完成，总耗时: {total_time:.2f} 秒")
    print("=" * 50)
    
    # 将所有结果合并到一个文件中
    print("\n保存综合结果...")
    
    # 创建一个包含所有策略结果的字典
    all_results = {
        '价值策略': value_stocks[['code', 'name']].assign(strategy='价值策略'),
        '成长策略': growth_stocks[['code', 'name']].assign(strategy='成长策略'),
        '质量策略': quality_stocks[['code', 'name']].assign(strategy='质量策略'),
        '动量策略': momentum_stocks[['code', 'name']].assign(strategy='动量策略'),
        '多因子策略': multi_factor_stocks[['code', 'name']].assign(strategy='多因子策略')
    }
    
    # 合并所有结果
    all_results_df = pd.concat(all_results.values())
    
    # 保存综合结果
    result_file = f'results/all_strategies_results_{datetime.now().strftime("%Y%m%d")}.csv'
    all_results_df.to_csv(result_file, index=False, encoding='utf-8-sig')
    print(f"综合结果已保存至: {result_file}")
    
    return {
        'value_stocks': value_stocks,
        'growth_stocks': growth_stocks,
        'quality_stocks': quality_stocks,
        'momentum_stocks': momentum_stocks,
        'multi_factor_stocks': multi_factor_stocks
    }

if __name__ == "__main__":
    # 运行所有策略
    results = run_all_strategies() 