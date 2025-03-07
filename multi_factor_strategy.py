#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多因子选股策略
综合价值、成长、质量和动量因子进行选股
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler

class MultiFactorStrategy:
    def __init__(self, 
                 value_weight=0.25,    # 价值因子权重
                 growth_weight=0.25,   # 成长因子权重
                 quality_weight=0.25,  # 质量因子权重
                 momentum_weight=0.25, # 动量因子权重
                 top_n=20):            # 最终选择的股票数量
        self.value_weight = value_weight
        self.growth_weight = growth_weight
        self.quality_weight = quality_weight
        self.momentum_weight = momentum_weight
        self.top_n = top_n
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_stock_list(self):
        """获取A股股票列表"""
        print("获取A股股票列表...")
        stock_info = ak.stock_info_a_code_name()
        return stock_info
    
    def get_value_factors(self):
        """获取价值因子数据"""
        print("获取价值因子数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 获取市盈率数据
        print("获取市盈率数据...")
        pe_df = ak.stock_a_pe()
        pe_df = pe_df.rename(columns={'代码': 'code', '市盈率': 'pe'})
        pe_df = pe_df[['code', 'pe']]
        
        # 获取市净率数据
        print("获取市净率数据...")
        pb_df = ak.stock_a_pb()
        pb_df = pb_df.rename(columns={'代码': 'code', '市净率': 'pb'})
        pb_df = pb_df[['code', 'pb']]
        
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
        value_df = pd.merge(stock_list, pe_df, on='code', how='left')
        value_df = pd.merge(value_df, pb_df, on='code', how='left')
        value_df = pd.merge(value_df, dividend_df, on='code', how='left')
        
        # 填充缺失值
        value_df = value_df.fillna({'pe': float('inf'), 'pb': float('inf'), 'dividend': 0})
        
        return value_df
    
    def get_growth_factors(self):
        """获取成长因子数据"""
        print("获取成长因子数据...")
        
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
        
        # 提取EPS增长率
        eps_growth_df = financial_data[['股票代码', '基本每股收益增长率(%)']]
        eps_growth_df = eps_growth_df.rename(columns={'股票代码': 'code', '基本每股收益增长率(%)': 'eps_growth'})
        eps_growth_df['eps_growth'] = eps_growth_df['eps_growth'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace('%', '')) if isinstance(x, str) and x != '--' else 0)
        
        # 合并数据
        growth_df = pd.merge(stock_list, profit_growth_df, on='code', how='left')
        growth_df = pd.merge(growth_df, revenue_growth_df, on='code', how='left')
        growth_df = pd.merge(growth_df, eps_growth_df, on='code', how='left')
        
        # 填充缺失值
        growth_df = growth_df.fillna({'profit_growth': 0, 'revenue_growth': 0, 'eps_growth': 0})
        
        return growth_df
    
    def get_quality_factors(self):
        """获取质量因子数据"""
        print("获取质量因子数据...")
        
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
        
        # 合并数据
        quality_df = pd.merge(stock_list, roe_df, on='code', how='left')
        quality_df = pd.merge(quality_df, gross_margin_df, on='code', how='left')
        quality_df = pd.merge(quality_df, debt_to_asset_df, on='code', how='left')
        
        # 填充缺失值
        quality_df = quality_df.fillna({'roe': 0, 'gross_margin': 0, 'debt_to_asset': 100})
        
        return quality_df
    
    def get_momentum_factors(self):
        """获取动量因子数据"""
        print("获取动量因子数据...")
        
        # 获取所有A股股票列表
        stock_list = self.get_stock_list()
        
        # 计算日期
        end_date = datetime.now()
        start_date_6m = (end_date - timedelta(days=180)).strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # 初始化结果DataFrame
        momentum_df = pd.DataFrame()
        momentum_df['code'] = stock_list['code']
        momentum_df['name'] = stock_list['name']
        momentum_df['momentum_1m'] = 0.0
        momentum_df['momentum_3m'] = 0.0
        
        # 获取股票历史数据并计算动量
        print("计算动量指标...")
        total_stocks = len(stock_list)
        
        # 为了演示，只处理前100只股票
        for i, (_, row) in enumerate(stock_list.head(100).iterrows()):
            code = row['code']
            print(f"处理 {code} ({i+1}/100)...")
            
            try:
                # 获取历史价格数据
                stock_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date_6m, end_date=end_date_str, adjust="qfq")
                
                if len(stock_data) > 0:
                    # 确保数据按日期排序
                    stock_data = stock_data.sort_values('日期')
                    
                    # 获取收盘价
                    close_prices = stock_data['收盘'].values
                    
                    # 计算动量（收益率）
                    if len(close_prices) >= 20:  # 至少需要20个交易日的数据
                        # 计算1个月动量
                        momentum_1m = (close_prices[-1] / close_prices[-20] - 1) * 100
                        momentum_df.loc[momentum_df['code'] == code, 'momentum_1m'] = momentum_1m
                        
                        # 计算3个月动量
                        if len(close_prices) >= 60:
                            momentum_3m = (close_prices[-1] / close_prices[-60] - 1) * 100
                            momentum_df.loc[momentum_df['code'] == code, 'momentum_3m'] = momentum_3m
            except Exception as e:
                print(f"处理 {code} 时出错: {e}")
        
        return momentum_df
    
    def calculate_factor_scores(self, data, factor_columns, ascending_list):
        """计算因子得分"""
        # 创建一个新的DataFrame来存储得分
        scores_df = pd.DataFrame(index=data.index)
        
        # 对每个因子进行标准化和排名
        for i, factor in enumerate(factor_columns):
            # 处理无穷大值
            data[factor] = data[factor].replace([np.inf, -np.inf], np.nan)
            
            # 填充缺失值
            if ascending_list[i]:
                # 如果是越小越好的因子，用最大值填充
                data[factor] = data[factor].fillna(data[factor].max())
            else:
                # 如果是越大越好的因子，用最小值填充
                data[factor] = data[factor].fillna(data[factor].min())
            
            # 标准化
            scaler = StandardScaler()
            data[f'{factor}_std'] = scaler.fit_transform(data[[factor]])
            
            # 计算排名得分
            scores_df[f'{factor}_score'] = data[factor].rank(ascending=ascending_list[i])
        
        return scores_df
    
    def select_stocks(self):
        """多因子选股"""
        print("开始多因子选股...")
        
        # 获取各类因子数据
        value_df = self.get_value_factors()
        growth_df = self.get_growth_factors()
        quality_df = self.get_quality_factors()
        momentum_df = self.get_momentum_factors()
        
        # 合并所有因子数据
        print("合并因子数据...")
        merged_df = pd.merge(value_df, growth_df[['code', 'profit_growth', 'revenue_growth', 'eps_growth']], on='code', how='left')
        merged_df = pd.merge(merged_df, quality_df[['code', 'roe', 'gross_margin', 'debt_to_asset']], on='code', how='left')
        merged_df = pd.merge(merged_df, momentum_df[['code', 'momentum_1m', 'momentum_3m']], on='code', how='left')
        
        # 填充缺失值
        merged_df = merged_df.fillna({
            'pe': float('inf'), 
            'pb': float('inf'), 
            'dividend': 0,
            'profit_growth': 0, 
            'revenue_growth': 0, 
            'eps_growth': 0,
            'roe': 0, 
            'gross_margin': 0, 
            'debt_to_asset': 100,
            'momentum_1m': 0, 
            'momentum_3m': 0
        })
        
        # 计算各类因子得分
        print("计算因子得分...")
        
        # 价值因子得分（PE和PB越小越好，股息率越大越好）
        value_scores = self.calculate_factor_scores(
            merged_df, 
            ['pe', 'pb', 'dividend'], 
            [True, True, False]  # PE和PB是越小越好，股息率是越大越好
        )
        
        # 成长因子得分（都是越大越好）
        growth_scores = self.calculate_factor_scores(
            merged_df, 
            ['profit_growth', 'revenue_growth', 'eps_growth'], 
            [False, False, False]  # 都是越大越好
        )
        
        # 质量因子得分（ROE和毛利率越大越好，资产负债率越小越好）
        quality_scores = self.calculate_factor_scores(
            merged_df, 
            ['roe', 'gross_margin', 'debt_to_asset'], 
            [False, False, True]  # ROE和毛利率是越大越好，资产负债率是越小越好
        )
        
        # 动量因子得分（都是越大越好）
        momentum_scores = self.calculate_factor_scores(
            merged_df, 
            ['momentum_1m', 'momentum_3m'], 
            [False, False]  # 都是越大越好
        )
        
        # 计算综合得分
        print("计算综合得分...")
        
        # 计算各类因子的平均得分
        merged_df['value_score'] = value_scores.mean(axis=1)
        merged_df['growth_score'] = growth_scores.mean(axis=1)
        merged_df['quality_score'] = quality_scores.mean(axis=1)
        merged_df['momentum_score'] = momentum_scores.mean(axis=1)
        
        # 计算加权综合得分
        merged_df['total_score'] = (
            merged_df['value_score'] * self.value_weight +
            merged_df['growth_score'] * self.growth_weight +
            merged_df['quality_score'] * self.quality_weight +
            merged_df['momentum_score'] * self.momentum_weight
        )
        
        # 按综合得分排序
        merged_df = merged_df.sort_values('total_score', ascending=False)
        
        # 选择前N只股票
        final_selection = merged_df.head(self.top_n)
        
        return final_selection
    
    def run(self):
        """运行策略并输出结果"""
        # 选股
        selected_stocks = self.select_stocks()
        
        # 输出结果
        print(f"\n===== 多因子选股结果 (前{self.top_n}只) =====")
        print(f"因子权重: 价值={self.value_weight}, 成长={self.growth_weight}, 质量={self.quality_weight}, 动量={self.momentum_weight}")
        print(selected_stocks[['code', 'name', 'value_score', 'growth_score', 'quality_score', 'momentum_score', 'total_score']])
        
        # 保存结果
        result_file = f'results/multi_factor_strategy_{datetime.now().strftime("%Y%m%d")}.csv'
        selected_stocks.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {result_file}")
        
        return selected_stocks

if __name__ == "__main__":
    # 创建策略实例
    strategy = MultiFactorStrategy(
        value_weight=0.25,
        growth_weight=0.25,
        quality_weight=0.25,
        momentum_weight=0.25,
        top_n=20
    )
    
    # 运行策略
    selected_stocks = strategy.run() 