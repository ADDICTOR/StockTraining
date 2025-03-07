#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双低双高价值选股策略
选股条件：低PE、低PB、高ROE、高股息
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ValueBasicStrategy:
    def __init__(self, 
                 pe_threshold=15,      # PE阈值，低于此值视为低PE
                 pb_threshold=1.5,     # PB阈值，低于此值视为低PB
                 roe_threshold=10,     # ROE阈值，高于此值视为高ROE
                 dividend_threshold=2, # 股息率阈值，高于此值视为高股息
                 top_n=20):            # 最终选择的股票数量
        self.pe_threshold = pe_threshold
        self.pb_threshold = pb_threshold
        self.roe_threshold = roe_threshold
        self.dividend_threshold = dividend_threshold
        self.top_n = top_n
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_stock_list(self):
        """获取A股股票列表"""
        print("获取A股股票列表...")
        stock_info = ak.stock_info_a_code_name()
        return stock_info
    
    def get_fundamental_data(self):
        """获取基本面数据"""
        print("获取基本面数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 获取最新市盈率、市净率数据
        print("获取市盈率数据...")
        pe_df = ak.stock_a_pe()
        pe_df = pe_df.rename(columns={'代码': 'code', '市盈率': 'pe'})
        pe_df = pe_df[['code', 'pe']]
        
        print("获取市净率数据...")
        pb_df = ak.stock_a_pb()
        pb_df = pb_df.rename(columns={'代码': 'code', '市净率': 'pb'})
        pb_df = pb_df[['code', 'pb']]
        
        # 获取ROE数据
        print("获取ROE数据...")
        # 使用最近一期财报数据
        roe_df = ak.stock_financial_analysis_indicator()
        roe_df = roe_df[['股票代码', '净资产收益率']]
        roe_df = roe_df.rename(columns={'股票代码': 'code', '净资产收益率': 'roe'})
        roe_df['roe'] = roe_df['roe'].apply(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x)
        
        # 获取股息率数据
        print("获取股息率数据...")
        dividend_df = ak.stock_history_dividend()
        # 计算最近一年的股息率
        dividend_df = dividend_df.groupby('代码').apply(lambda x: x.sort_values('除权除息日', ascending=False).head(1))
        dividend_df = dividend_df.reset_index(drop=True)
        dividend_df = dividend_df[['代码', '股息率']]
        dividend_df = dividend_df.rename(columns={'代码': 'code', '股息率': 'dividend'})
        dividend_df['dividend'] = dividend_df['dividend'].apply(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x)
        
        # 合并数据
        print("合并数据...")
        merged_df = pd.merge(stock_list, pe_df, left_on='code', right_on='code', how='left')
        merged_df = pd.merge(merged_df, pb_df, left_on='code', right_on='code', how='left')
        merged_df = pd.merge(merged_df, roe_df, left_on='code', right_on='code', how='left')
        merged_df = pd.merge(merged_df, dividend_df, left_on='code', right_on='code', how='left')
        
        # 填充缺失值
        merged_df = merged_df.fillna({'pe': float('inf'), 'pb': float('inf'), 'roe': 0, 'dividend': 0})
        
        return merged_df
    
    def select_stocks(self):
        """根据双低双高策略选股"""
        print("开始选股...")
        
        # 获取基本面数据
        data = self.get_fundamental_data()
        
        # 应用选股条件
        selected = data[
            (data['pe'] < self.pe_threshold) & 
            (data['pb'] < self.pb_threshold) & 
            (data['roe'] > self.roe_threshold) & 
            (data['dividend'] > self.dividend_threshold)
        ]
        
        # 计算综合得分 (低PE和低PB是好的，高ROE和高股息是好的)
        selected['pe_score'] = selected['pe'].rank(ascending=True)
        selected['pb_score'] = selected['pb'].rank(ascending=True)
        selected['roe_score'] = selected['roe'].rank(ascending=False)
        selected['dividend_score'] = selected['dividend'].rank(ascending=False)
        
        # 计算总分
        selected['total_score'] = selected['pe_score'] + selected['pb_score'] + selected['roe_score'] + selected['dividend_score']
        
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
        print(f"\n===== 双低双高价值选股结果 (前{self.top_n}只) =====")
        print(f"选股条件: PE < {self.pe_threshold}, PB < {self.pb_threshold}, ROE > {self.roe_threshold}%, 股息率 > {self.dividend_threshold}%")
        print(selected_stocks[['code', 'name', 'pe', 'pb', 'roe', 'dividend', 'total_score']])
        
        # 保存结果
        result_file = f'results/value_basic_strategy_{datetime.now().strftime("%Y%m%d")}.csv'
        selected_stocks.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {result_file}")
        
        return selected_stocks

if __name__ == "__main__":
    # 创建策略实例
    strategy = ValueBasicStrategy(
        pe_threshold=15,
        pb_threshold=1.5,
        roe_threshold=10,
        dividend_threshold=2,
        top_n=20
    )
    
    # 运行策略
    selected_stocks = strategy.run() 