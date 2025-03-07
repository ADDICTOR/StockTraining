#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
质量型选股策略
选股条件：高ROE、高毛利率、低负债率、高现金流质量
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class QualityStrategy:
    def __init__(self, 
                 roe_threshold=15,                # ROE阈值
                 gross_margin_threshold=30,       # 毛利率阈值
                 debt_to_asset_threshold=50,      # 资产负债率阈值（低于此值）
                 cash_flow_quality_threshold=1.0, # 现金流质量阈值（经营现金流/净利润）
                 top_n=20):                       # 最终选择的股票数量
        self.roe_threshold = roe_threshold
        self.gross_margin_threshold = gross_margin_threshold
        self.debt_to_asset_threshold = debt_to_asset_threshold
        self.cash_flow_quality_threshold = cash_flow_quality_threshold
        self.top_n = top_n
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_stock_list(self):
        """获取A股股票列表"""
        print("获取A股股票列表...")
        stock_info = ak.stock_info_a_code_name()
        return stock_info
    
    def get_quality_data(self):
        """获取质量指标数据"""
        print("获取质量指标数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 获取财务指标数据
        print("获取财务指标数据...")
        financial_data = ak.stock_financial_analysis_indicator()
        
        # 提取ROE
        roe_df = financial_data[['股票代码', '净资产收益率(%)']]
        roe_df = roe_df.rename(columns={'股票代码': 'code', '净资产收益率(%)': 'roe'})
        roe_df['roe'] = roe_df['roe'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 提取毛利率
        gross_margin_df = financial_data[['股票代码', '销售毛利率(%)']]
        gross_margin_df = gross_margin_df.rename(columns={'股票代码': 'code', '销售毛利率(%)': 'gross_margin'})
        gross_margin_df['gross_margin'] = gross_margin_df['gross_margin'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 提取资产负债率
        debt_to_asset_df = financial_data[['股票代码', '资产负债率(%)']]
        debt_to_asset_df = debt_to_asset_df.rename(columns={'股票代码': 'code', '资产负债率(%)': 'debt_to_asset'})
        debt_to_asset_df['debt_to_asset'] = debt_to_asset_df['debt_to_asset'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 获取现金流数据
        print("获取现金流数据...")
        # 获取最新财报的经营活动现金流和净利润
        cash_flow_df = financial_data[['股票代码', '经营活动产生的现金流量净额', '净利润']]
        cash_flow_df = cash_flow_df.rename(columns={'股票代码': 'code', '经营活动产生的现金流量净额': 'operating_cash_flow', '净利润': 'net_profit'})
        
        # 转换为数值类型
        cash_flow_df['operating_cash_flow'] = cash_flow_df['operating_cash_flow'].apply(lambda x: float(x) if isinstance(x, (int, float)) else 0)
        cash_flow_df['net_profit'] = cash_flow_df['net_profit'].apply(lambda x: float(x) if isinstance(x, (int, float)) else 0)
        
        # 计算现金流质量（经营现金流/净利润）
        cash_flow_df['cash_flow_quality'] = cash_flow_df.apply(
            lambda row: row['operating_cash_flow'] / row['net_profit'] if row['net_profit'] > 0 else 0, 
            axis=1
        )
        
        # 合并数据
        print("合并数据...")
        merged_df = pd.merge(stock_list, roe_df, on='code', how='left')
        merged_df = pd.merge(merged_df, gross_margin_df, on='code', how='left')
        merged_df = pd.merge(merged_df, debt_to_asset_df, on='code', how='left')
        merged_df = pd.merge(merged_df, cash_flow_df[['code', 'cash_flow_quality']], on='code', how='left')
        
        # 填充缺失值
        merged_df = merged_df.fillna({
            'roe': 0, 
            'gross_margin': 0, 
            'debt_to_asset': 100,  # 默认为最高负债率
            'cash_flow_quality': 0
        })
        
        return merged_df
    
    def select_stocks(self):
        """根据质量指标选股"""
        print("开始选股...")
        
        # 获取质量指标数据
        data = self.get_quality_data()
        
        # 应用选股条件
        selected = data[
            (data['roe'] > self.roe_threshold) & 
            (data['gross_margin'] > self.gross_margin_threshold) & 
            (data['debt_to_asset'] < self.debt_to_asset_threshold) & 
            (data['cash_flow_quality'] > self.cash_flow_quality_threshold)
        ]
        
        # 计算综合得分
        selected['roe_score'] = selected['roe'].rank(ascending=False)
        selected['gross_margin_score'] = selected['gross_margin'].rank(ascending=False)
        selected['debt_to_asset_score'] = selected['debt_to_asset'].rank(ascending=True)  # 负债率越低越好
        selected['cash_flow_quality_score'] = selected['cash_flow_quality'].rank(ascending=False)
        
        # 计算总分
        selected['total_score'] = (
            selected['roe_score'] + 
            selected['gross_margin_score'] + 
            selected['debt_to_asset_score'] + 
            selected['cash_flow_quality_score']
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
        print(f"\n===== 质量型选股结果 (前{self.top_n}只) =====")
        print(f"选股条件: ROE > {self.roe_threshold}%, 毛利率 > {self.gross_margin_threshold}%, 资产负债率 < {self.debt_to_asset_threshold}%, 现金流质量 > {self.cash_flow_quality_threshold}")
        print(selected_stocks[['code', 'name', 'roe', 'gross_margin', 'debt_to_asset', 'cash_flow_quality', 'total_score']])
        
        # 保存结果
        result_file = f'results/quality_strategy_{datetime.now().strftime("%Y%m%d")}.csv'
        selected_stocks.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {result_file}")
        
        return selected_stocks

if __name__ == "__main__":
    # 创建策略实例
    strategy = QualityStrategy(
        roe_threshold=15,
        gross_margin_threshold=30,
        debt_to_asset_threshold=50,
        cash_flow_quality_threshold=1.0,
        top_n=20
    )
    
    # 运行策略
    selected_stocks = strategy.run() 