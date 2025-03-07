#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成长型选股策略
选股条件：高净利润增长率、高营收增长率、高ROE增长率、高EPS增长率
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class GrowthStrategy:
    def __init__(self, 
                 profit_growth_threshold=20,    # 净利润增长率阈值
                 revenue_growth_threshold=15,   # 营收增长率阈值
                 roe_growth_threshold=5,        # ROE增长率阈值
                 eps_growth_threshold=15,       # EPS增长率阈值
                 top_n=20):                     # 最终选择的股票数量
        self.profit_growth_threshold = profit_growth_threshold
        self.revenue_growth_threshold = revenue_growth_threshold
        self.roe_growth_threshold = roe_growth_threshold
        self.eps_growth_threshold = eps_growth_threshold
        self.top_n = top_n
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_stock_list(self):
        """获取A股股票列表"""
        print("获取A股股票列表...")
        stock_info = ak.stock_info_a_code_name()
        return stock_info
    
    def get_growth_data(self):
        """获取成长性数据"""
        print("获取成长性数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 获取财务指标数据
        print("获取财务指标数据...")
        financial_data = ak.stock_financial_analysis_indicator()
        
        # 提取净利润增长率
        profit_growth_df = financial_data[['股票代码', '净利润增长率(%)']]
        profit_growth_df = profit_growth_df.rename(columns={'股票代码': 'code', '净利润增长率(%)': 'profit_growth'})
        profit_growth_df['profit_growth'] = profit_growth_df['profit_growth'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 提取营业收入增长率
        revenue_growth_df = financial_data[['股票代码', '营业收入增长率(%)']]
        revenue_growth_df = revenue_growth_df.rename(columns={'股票代码': 'code', '营业收入增长率(%)': 'revenue_growth'})
        revenue_growth_df['revenue_growth'] = revenue_growth_df['revenue_growth'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 提取ROE增长率 (需要计算)
        # 获取最近两年的ROE数据
        roe_df = financial_data[['股票代码', '净资产收益率(%)', '上年度净资产收益率(%)']]
        roe_df = roe_df.rename(columns={'股票代码': 'code', '净资产收益率(%)': 'roe_current', '上年度净资产收益率(%)': 'roe_previous'})
        
        # 转换为数值类型
        roe_df['roe_current'] = roe_df['roe_current'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        roe_df['roe_previous'] = roe_df['roe_previous'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 计算ROE增长率
        roe_df['roe_growth'] = roe_df.apply(
            lambda row: ((row['roe_current'] - row['roe_previous']) / abs(row['roe_previous']) * 100) if row['roe_previous'] != 0 else 0, 
            axis=1
        )
        
        # 提取EPS增长率
        eps_growth_df = financial_data[['股票代码', '基本每股收益增长率(%)']]
        eps_growth_df = eps_growth_df.rename(columns={'股票代码': 'code', '基本每股收益增长率(%)': 'eps_growth'})
        eps_growth_df['eps_growth'] = eps_growth_df['eps_growth'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 合并数据
        print("合并数据...")
        merged_df = pd.merge(stock_list, profit_growth_df, on='code', how='left')
        merged_df = pd.merge(merged_df, revenue_growth_df, on='code', how='left')
        merged_df = pd.merge(merged_df, roe_df[['code', 'roe_growth']], on='code', how='left')
        merged_df = pd.merge(merged_df, eps_growth_df, on='code', how='left')
        
        # 填充缺失值
        merged_df = merged_df.fillna({
            'profit_growth': 0, 
            'revenue_growth': 0, 
            'roe_growth': 0, 
            'eps_growth': 0
        })
        
        return merged_df
    
    def select_stocks(self):
        """根据成长性指标选股"""
        print("开始选股...")
        
        # 获取成长性数据
        data = self.get_growth_data()
        
        # 应用选股条件
        selected = data[
            (data['profit_growth'] > self.profit_growth_threshold) & 
            (data['revenue_growth'] > self.revenue_growth_threshold) & 
            (data['roe_growth'] > self.roe_growth_threshold) & 
            (data['eps_growth'] > self.eps_growth_threshold)
        ]
        
        # 计算综合得分 (所有指标都是越高越好)
        selected['profit_growth_score'] = selected['profit_growth'].rank(ascending=False)
        selected['revenue_growth_score'] = selected['revenue_growth'].rank(ascending=False)
        selected['roe_growth_score'] = selected['roe_growth'].rank(ascending=False)
        selected['eps_growth_score'] = selected['eps_growth'].rank(ascending=False)
        
        # 计算总分
        selected['total_score'] = (
            selected['profit_growth_score'] + 
            selected['revenue_growth_score'] + 
            selected['roe_growth_score'] + 
            selected['eps_growth_score']
        )
        
        # 按总分排序
        selected = selected.sort_values('total_score', ascending=False)
        
        # 选择前N只股票
        final_selection = selected.head(self.top_n)
        
        return final_selection
    
    def run(self):
        """运行策略并输出结果"""
        # 选股
        selected_stocks = self.select_stocks()
        
        # 输出结果
        print(f"\n===== 成长型选股结果 (前{self.top_n}只) =====")
        print(f"选股条件: 净利润增长率 > {self.profit_growth_threshold}%, 营收增长率 > {self.revenue_growth_threshold}%, ROE增长率 > {self.roe_growth_threshold}%, EPS增长率 > {self.eps_growth_threshold}%")
        print(selected_stocks[['code', 'name', 'profit_growth', 'revenue_growth', 'roe_growth', 'eps_growth', 'total_score']])
        
        # 保存结果
        result_file = f'results/growth_strategy_{datetime.now().strftime("%Y%m%d")}.csv'
        selected_stocks.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {result_file}")
        
        return selected_stocks

if __name__ == "__main__":
    # 创建策略实例
    strategy = GrowthStrategy(
        profit_growth_threshold=20,
        revenue_growth_threshold=15,
        roe_growth_threshold=5,
        eps_growth_threshold=15,
        top_n=20
    )
    
    # 运行策略
    selected_stocks = strategy.run() 