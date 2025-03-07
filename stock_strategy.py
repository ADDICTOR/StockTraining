#!/usr/bin/env python
# -*- coding: utf-8 -*-

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PortfolioBacktest:
    def __init__(self):
        # 初始化资产配置比例
        self.initial_allocation = {
            '中证500': 0.35,
            '标普500': 0.15,
            '红利': 0.10,
            '黄金': 0.10,
            '消费': 0.15,
            '科创50': 0.15
        }
        
        # 初始化数据存储
        self.data = {}
        self.start_date = '2013-01-01'  # 近十年数据
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 指数与代码映射及获取方法
        self.index_info = {
            '中证500': {'symbol': 'sh000905', 'func': self.get_index_data},
            '标普500': {'symbol': 'sh000001', 'func': self.get_index_data},  # 用上证指数代替标普500
            '红利': {'symbol': 'sh000015', 'func': self.get_index_data},  # 上证红利指数
            '黄金': {'symbol': 'Au(T+D)', 'func': self.get_gold_data},  # 黄金9999
            '消费': {'symbol': 'sh000932', 'func': self.get_index_data},  # 中证800消费
            '科创50': {'symbol': 'sh000688', 'func': self.get_index_data}  # 科创50
        }
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def get_index_data(self, symbol):
        """获取指数数据"""
        try:
            # 使用 stock_zh_index_daily_em 获取指数数据
            print(f"正在获取 {symbol} 的历史数据...")
            df = ak.stock_zh_index_daily_em(symbol=symbol, start_date=self.start_date.replace('-', ''), end_date=self.end_date.replace('-', ''))
            
            # 重命名列
            df = df.rename(columns={'date': 'date', 'close': 'close'})
            
            # 确保日期列是日期类型
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 只保留收盘价列
            df = df[['close']]
            
            print(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def get_gold_data(self, symbol):
        """获取黄金数据"""
        try:
            # 使用 spot_hist_sge 获取黄金数据
            print(f"正在获取黄金 {symbol} 的历史数据...")
            df = ak.spot_hist_sge(symbol=symbol)
            
            # 确保日期列是日期类型
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 只保留收盘价列
            df = df[['close']]
            
            # 确保数据在指定日期范围内
            df = df.loc[self.start_date:self.end_date]
            
            print(f"成功获取黄金数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"获取黄金数据失败: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def fetch_data(self):
        """获取各指数的历史数据"""
        print("正在获取数据...")
        
        for name, info in self.index_info.items():
            try:
                # 使用对应的函数获取数据
                df = info['func'](info['symbol'])
                
                # 计算每日收益率
                df['daily_return'] = df['close'].pct_change()
                
                self.data[name] = df
                print(f"成功获取 {name} 数据，共 {len(df)} 条记录")
            except Exception as e:
                print(f"获取 {name} 数据失败: {e}")
                import traceback
                print(traceback.format_exc())
        
        # 检查是否有足够的数据继续
        if not self.data:
            raise ValueError("没有成功获取任何数据，无法继续回测")
        elif len(self.data) < len(self.initial_allocation):
            print(f"警告: 只获取到 {len(self.data)}/{len(self.initial_allocation)} 个资产的数据")
            print(f"已获取数据的资产: {list(self.data.keys())}")
            print("将使用可用数据继续回测，但结果可能不准确")
            
            # 重新调整资产配置比例
            available_assets = list(self.data.keys())
            total_weight = sum(self.initial_allocation[asset] for asset in available_assets)
            
            adjusted_allocation = {}
            for asset in available_assets:
                adjusted_allocation[asset] = self.initial_allocation[asset] / total_weight
            
            self.initial_allocation = adjusted_allocation
            
            print("调整后的资产配置比例:")
            for asset, weight in self.initial_allocation.items():
                print(f"{asset}: {weight:.2f}")
        
        # 对齐所有数据的日期
        self.align_dates()
    
    def align_dates(self):
        """确保所有数据使用相同的日期索引"""
        # 找出所有数据集共有的日期
        common_dates = None
        
        for name, df in self.data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # 重新索引所有数据集
        for name in self.data:
            self.data[name] = self.data[name].loc[common_dates]
        
        print(f"数据对齐完成，共有 {len(common_dates)} 个交易日")
    
    def run_backtest(self, rebalance_threshold=None):
        """
        运行回测
        
        参数:
        rebalance_threshold: 触发再平衡的阈值，如0.2表示当任一资产配置偏离目标20%时进行再平衡
        """
        # 初始化投资组合
        initial_investment = 1000000  # 初始投资100万
        portfolio_value = initial_investment
        
        # 计算初始持仓
        holdings = {}
        for asset, allocation in self.initial_allocation.items():
            if asset in self.data:  # 只使用可用的资产数据
                holdings[asset] = portfolio_value * allocation / self.data[asset]['close'].iloc[0]
        
        # 记录每日投资组合价值
        portfolio_history = []
        rebalance_dates = []
        
        # 获取共同的日期索引
        dates = self.data[list(self.data.keys())[0]].index
        
        for i, date in enumerate(dates):
            if i == 0:
                portfolio_history.append(portfolio_value)
                continue
            
            # 计算当前投资组合价值
            current_value = 0
            asset_values = {}
            
            for asset, shares in holdings.items():
                asset_value = shares * self.data[asset]['close'].loc[date]
                asset_values[asset] = asset_value
                current_value += asset_value
            
            # 检查是否需要再平衡
            need_rebalance = False
            if rebalance_threshold is not None:
                for asset, allocation in self.initial_allocation.items():
                    if asset in self.data:  # 只检查可用的资产
                        current_allocation = asset_values[asset] / current_value
                        # 如果任一资产配置偏离目标超过阈值，则需要再平衡
                        if abs(current_allocation - allocation) / allocation > rebalance_threshold:
                            need_rebalance = True
                            break
            
            # 执行再平衡
            if need_rebalance:
                rebalance_dates.append(date)
                for asset, allocation in self.initial_allocation.items():
                    if asset in self.data:  # 只再平衡可用的资产
                        target_value = current_value * allocation
                        holdings[asset] = target_value / self.data[asset]['close'].loc[date]
            
            portfolio_history.append(current_value)
        
        # 计算投资组合的每日收益率
        portfolio_returns = pd.Series(portfolio_history, index=dates).pct_change()
        
        # 计算累计收益率
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # 计算年化收益率
        years = (dates[-1] - dates[0]).days / 365
        annualized_return = (portfolio_history[-1] / portfolio_history[0]) ** (1 / years) - 1
        
        # 计算最大回撤
        max_drawdown = 0
        peak = portfolio_history[0]
        
        for value in portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))
        
        # 返回回测结果
        return {
            'portfolio_history': pd.Series(portfolio_history, index=dates),
            'cumulative_returns': cumulative_returns,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'rebalance_dates': rebalance_dates,
            'rebalance_count': len(rebalance_dates)
        }
    
    def plot_results(self, results, rebalance_threshold=None):
        """绘制回测结果图表"""
        plt.figure(figsize=(14, 10))
        
        # 绘制投资组合价值曲线
        plt.subplot(2, 1, 1)
        plt.plot(results['portfolio_history'])
        plt.title('投资组合价值变化')
        plt.xlabel('日期')
        plt.ylabel('价值 (元)')
        plt.grid(True)
        
        # 标记再平衡点
        if rebalance_threshold is not None and results['rebalance_dates']:
            rebalance_values = [results['portfolio_history'][date] for date in results['rebalance_dates']]
            plt.scatter(results['rebalance_dates'], rebalance_values, color='red', s=30, label='再平衡点')
            plt.legend()
        
        # 绘制累计收益率曲线
        plt.subplot(2, 1, 2)
        plt.plot(results['cumulative_returns'] * 100)
        plt.title('累计收益率 (%)')
        plt.xlabel('日期')
        plt.ylabel('累计收益率 (%)')
        plt.grid(True)
        
        # 添加性能指标文本
        plt.figtext(0.15, 0.02, f'年化收益率: {results["annualized_return"]*100:.2f}%', fontsize=12)
        plt.figtext(0.35, 0.02, f'最大回撤: {results["max_drawdown"]*100:.2f}%', fontsize=12)
        plt.figtext(0.55, 0.02, f'夏普比率: {results["sharpe_ratio"]:.2f}', fontsize=12)
        
        if rebalance_threshold is not None:
            plt.figtext(0.75, 0.02, f'再平衡次数: {results["rebalance_count"]}', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        filename = f'results/portfolio_backtest{"_rebalance_" + str(int(rebalance_threshold*100)) if rebalance_threshold else ""}.png'
        plt.savefig(filename)
        plt.close()
        
        print(f"结果图表已保存至 {filename}")
    
    def run(self):
        """运行完整的回测流程"""
        # 获取数据
        self.fetch_data()
        
        # 运行不再平衡的回测
        print("\n运行不再平衡的回测...")
        no_rebalance_results = self.run_backtest()
        self.plot_results(no_rebalance_results)
        
        # 运行20%阈值再平衡的回测
        print("\n运行20%阈值再平衡的回测...")
        rebalance_results = self.run_backtest(rebalance_threshold=0.03)
        self.plot_results(rebalance_results, rebalance_threshold=0.03)
        
        # 打印比较结果
        print("\n===== 回测结果比较 =====")
        print(f"{'策略':^15}{'年化收益率':^15}{'最大回撤':^15}{'夏普比率':^15}{'再平衡次数':^15}")
        print(f"{'不再平衡':^15}{no_rebalance_results['annualized_return']*100:^15.2f}%{no_rebalance_results['max_drawdown']*100:^15.2f}%{no_rebalance_results['sharpe_ratio']:^15.2f}{0:^15}")
        print(f"{'20%阈值再平衡':^15}{rebalance_results['annualized_return']*100:^15.2f}%{rebalance_results['max_drawdown']*100:^15.2f}%{rebalance_results['sharpe_ratio']:^15.2f}{rebalance_results['rebalance_count']:^15}")
        
        # 计算改进百分比
        return_improvement = (rebalance_results['annualized_return'] - no_rebalance_results['annualized_return']) / abs(no_rebalance_results['annualized_return']) * 100
        drawdown_improvement = (no_rebalance_results['max_drawdown'] - rebalance_results['max_drawdown']) / no_rebalance_results['max_drawdown'] * 100
        sharpe_improvement = (rebalance_results['sharpe_ratio'] - no_rebalance_results['sharpe_ratio']) / abs(no_rebalance_results['sharpe_ratio']) * 100
        
        print("\n===== 改进百分比 =====")
        print(f"年化收益率改进: {return_improvement:.2f}%")
        print(f"最大回撤改进: {drawdown_improvement:.2f}%")
        print(f"夏普比率改进: {sharpe_improvement:.2f}%")
        
        return {
            'no_rebalance': no_rebalance_results,
            'rebalance': rebalance_results
        }


if __name__ == "__main__":
    backtest = PortfolioBacktest()
    backtest.run() 