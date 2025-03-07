#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量型选股策略
选股条件：基于过去3个月、6个月和12个月的价格动量
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class MomentumStrategy:
    def __init__(self, 
                 momentum_3m_weight=0.5,  # 3个月动量权重
                 momentum_6m_weight=0.3,  # 6个月动量权重
                 momentum_12m_weight=0.2, # 12个月动量权重
                 top_n=20):              # 最终选择的股票数量
        self.momentum_3m_weight = momentum_3m_weight
        self.momentum_6m_weight = momentum_6m_weight
        self.momentum_12m_weight = momentum_12m_weight
        self.top_n = top_n
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_stock_list(self):
        """获取A股股票列表"""
        print("获取A股股票列表...")
        stock_info = ak.stock_info_a_code_name()
        return stock_info
    
    def get_momentum_data(self):
        """获取动量数据"""
        print("获取动量数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 计算日期
        end_date = datetime.now()
        start_date_12m = (end_date - timedelta(days=365)).strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame()
        result_df['code'] = stock_list['code']
        result_df['name'] = stock_list['name']
        result_df['momentum_3m'] = 0.0
        result_df['momentum_6m'] = 0.0
        result_df['momentum_12m'] = 0.0
        
        # 获取股票历史数据并计算动量
        print("计算动量指标...")
        total_stocks = len(stock_list)
        
        for i, (_, row) in enumerate(stock_list.iterrows()):
            code = row['code']
            print(f"处理 {code} ({i+1}/{total_stocks})...")
            
            try:
                # 获取历史价格数据
                if code.startswith('6'):
                    stock_code = f"sh{code}"
                else:
                    stock_code = f"sz{code}"
                
                # 使用akshare获取股票历史数据
                stock_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date_12m, end_date=end_date_str, adjust="qfq")
                
                if len(stock_data) > 0:
                    # 确保数据按日期排序
                    stock_data = stock_data.sort_values('日期')
                    
                    # 获取收盘价
                    close_prices = stock_data['收盘'].values
                    
                    # 计算动量（收益率）
                    if len(close_prices) >= 60:  # 至少需要60个交易日的数据
                        # 计算3个月动量
                        momentum_3m = (close_prices[-1] / close_prices[-60] - 1) * 100
                        result_df.loc[result_df['code'] == code, 'momentum_3m'] = momentum_3m
                        
                        # 计算6个月动量
                        if len(close_prices) >= 120:
                            momentum_6m = (close_prices[-1] / close_prices[-120] - 1) * 100
                            result_df.loc[result_df['code'] == code, 'momentum_6m'] = momentum_6m
                        
                        # 计算12个月动量
                        if len(close_prices) >= 240:
                            momentum_12m = (close_prices[-1] / close_prices[-240] - 1) * 100
                            result_df.loc[result_df['code'] == code, 'momentum_12m'] = momentum_12m
            except Exception as e:
                print(f"处理 {code} 时出错: {e}")
        
        return result_df
    
    def select_stocks(self):
        """根据动量指标选股"""
        print("开始选股...")
        
        # 获取动量数据
        data = self.get_momentum_data()
        
        # 过滤掉没有足够数据的股票
        data = data[(data['momentum_3m'] != 0) & (data['momentum_6m'] != 0) & (data['momentum_12m'] != 0)]
        
        # 计算综合动量得分
        data['momentum_score'] = (
            data['momentum_3m'] * self.momentum_3m_weight + 
            data['momentum_6m'] * self.momentum_6m_weight + 
            data['momentum_12m'] * self.momentum_12m_weight
        )
        
        # 按综合动量得分排序
        data = data.sort_values('momentum_score', ascending=False)
        
        # 选择前N只股票
        final_selection = data.head(self.top_n)
        
        return final_selection
    
    def run(self):
        """运行策略并输出结果"""
        # 选股
        selected_stocks = self.select_stocks()
        
        # 输出结果
        print(f"\n===== 动量型选股结果 (前{self.top_n}只) =====")
        print(f"选股条件: 3个月动量权重 = {self.momentum_3m_weight}, 6个月动量权重 = {self.momentum_6m_weight}, 12个月动量权重 = {self.momentum_12m_weight}")
        print(selected_stocks[['code', 'name', 'momentum_3m', 'momentum_6m', 'momentum_12m', 'momentum_score']])
        
        # 保存结果
        result_file = f'results/momentum_strategy_{datetime.now().strftime("%Y%m%d")}.csv'
        selected_stocks.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {result_file}")
        
        return selected_stocks

if __name__ == "__main__":
    # 创建策略实例
    strategy = MomentumStrategy(
        momentum_3m_weight=0.5,
        momentum_6m_weight=0.3,
        momentum_12m_weight=0.2,
        top_n=20
    )
    
    # 运行策略
    selected_stocks = strategy.run() 