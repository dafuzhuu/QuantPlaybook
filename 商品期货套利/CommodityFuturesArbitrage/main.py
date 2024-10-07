from causis_api.const import get_version
from causis_api.const import login
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
login.username = 'dafu.zhu'
login.password = '123456'
login.version = get_version()
from causis_api.data import *
import pandas as pd
import numpy as np

zh_font1 = matplotlib.font_manager.FontProperties(fname=r"font/SourceHanSansSC-Normal.otf")


class CommodityFutureArbitrage:
    """
    主要参数：止损线、开仓条件（多少sigma）
    """
    def __init__(self, contracts, proportion):
        self.contracts = contracts  # 拉取数据
        self.proportion = proportion   # 原料产出配比
        self.names = [name.split('.')[3] for name in self.contracts]
        self.price_df = pd.DataFrame()
        # 当前的做空、做多或空仓状态
        self.status = 0
        self.position = pd.DataFrame()
        self.current_cash = 1000000    # 初始化总资金100万
        self.contracts_value = 0    # 期货合约价值
        self.values = []     # 总资产随时间变化
        self.alpha = 0   # 交易几份spread
        self.alphas = []  # 仓位变化情况，用于画仓位图
        self.risk_day = 0   # 最近一次止损
        self.sensitivity_record = []

    # 获取数据
    def fetch_data(self):
        for contract in self.contracts:
            tmp_df = get_price(contract, start_date='2013-3-13')[['CLOCK', 'CLOSE']]
            tmp_df.set_index('CLOCK', inplace=True)
            self.price_df = tmp_df if self.price_df.empty else pd.merge(self.price_df, tmp_df, on='CLOCK')

        self.price_df.columns = self.names

    # 计算开仓平仓信号
    def calculate_signals(self, window=21, close=0.8, distal=1.0):
        # 利润回归模型
        # 计算价差
        self.price_df['spread'] = sum(x * self.price_df[y] for x, y in zip(self.proportion, self.names))
        self.price_df['mean'] = self.price_df['spread'].rolling(window=window).mean()
        self.price_df['std'] = self.price_df['spread'].rolling(window=window).std()
        self.price_df['lower_short'] = self.price_df['mean'] + close * self.price_df['std']
        self.price_df['upper_short'] = self.price_df['mean'] + distal * self.price_df['std']
        self.price_df['lower_long'] = self.price_df['mean'] - distal * self.price_df['std']
        self.price_df['upper_long'] = self.price_df['mean'] - close * self.price_df['std']
        self.price_df = self.price_df.dropna()
        self.position = self.price_df.copy().iloc[:, :len(self.names)]
        self.position.iloc[:, :len(self.names)] = 0

    # 开始交易
    def trade(self):
        """
        总共4种情况：未开仓->开仓；开仓->平仓；未开仓->未开仓；开仓->开仓
        :return:
        """
        df = self.price_df
        s0 = 0   # 初始化上一期的spread
        s0_bar = 0   # 初始化上一期spread对应的前10天mean
        init_position = -1 * np.array(self.proportion)   # 空首位
        for idx, row in df.iterrows():
            s = row['spread']
            s_bar = row['mean']

            # 开仓条件1: 未开仓且不在止损后10天内
            if (self.status == 0) and (df.index.get_loc(idx) - self.risk_day > 10 or self.risk_day == 0):
                # 合约金额为总金额的30%
                self.alpha = 0.3 * self.current_cash / s

                # 前两个情况为开仓，确认方向
                if row['lower_short'] < s < row['upper_short'] and s < s0:
                    # print("Entering a short position")
                    self.position.loc[idx] = self.alpha * init_position
                    self.status = 1
                    # 构造合约组合支付的费用
                    self.contracts_value = np.sum(np.array(row[self.names]) * self.position.loc[idx])
                    # 现金减去支付费用为留存现金
                    self.current_cash -= self.contracts_value
                    self.alphas.append(-self.alpha)    # 负数表示做空螺纹钢

                elif row['lower_long'] < s < row['upper_long'] and s > s0:
                    # print("Entering a long position")
                    self.position.loc[idx] = -self.alpha * init_position
                    self.status = 1
                    # 构造合约组合支付的费用
                    self.contracts_value = np.sum(np.array(row[self.names]) * self.position.loc[idx])
                    # 现金减去支付费用为留存现金
                    self.current_cash -= self.contracts_value
                    self.alphas.append(self.alpha)

                # 后一个情况为保持未开仓状态，不做操作
                else:
                    self.alphas.append(0)
                    pass

                self.values.append(self.current_cash + self.contracts_value)

            # 平仓条件1: 已开仓
            elif self.status:
                # 上一期头寸配置
                previous_position = self.position.shift(1).loc[idx]
                # 开仓->平仓
                if (s - s_bar) * (s0 - s0_bar) < 0:   # 穿过MA线
                    # 抹平头寸
                    self.position.loc[idx] = 0 * init_position
                    self.status = 0
                    # 清点平仓时合约组合收益，用上期头寸数据
                    self.current_cash += np.sum(np.array(row[self.names]) * previous_position)
                    # 合约价值清零
                    self.contracts_value = 0
                    self.values.append(self.current_cash + self.contracts_value)
                    self.alphas.append(0)

                # 保持开仓
                else:
                    # 更新合约组合价值，头寸不做改动
                    self.position.loc[idx] = previous_position
                    self.contracts_value = np.sum(np.array(row[self.names]) * self.position.loc[idx])
                    self.values.append(self.current_cash + self.contracts_value)
                    self.alphas.append(self.alphas[-1])
                    # 止损指令
                    if self.values[-1] / self.values[-2] - 1 < -0.02:
                        self.position.loc[idx] = 0 * init_position
                        self.risk_day = df.index.get_loc(idx)
                        self.status = 0

            # 未开仓，但在risk_day的10天内
            else:
                self.values.append(self.values[-1])
                self.alphas.append(0)

            s0 = s
            s0_bar = s_bar

    # 存储结果
    def save_data(self, file_name):
        self.price_df.to_excel(f'results/price_df_{file_name}.xlsx')
        self.position.to_excel(f'results/position_{file_name}.xlsx')
        value = pd.DataFrame(self.values, index=self.price_df.index)
        value.to_excel(f'results/values_{file_name}.xlsx')

    # 仓位柱状图，Pnl曲线
    def pnl(self, title):
        # 生成示例数据
        dates = self.price_df.index.tolist()
        positions = self.alphas  # 仓位柱状图
        pnl = np.array(self.values) / self.values[0]  # Pnl曲线

        fig, ax1 = plt.subplots(figsize=(15, 6))

        # PnL曲线，现在将其设置为左侧Y轴
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.plot(pnl, color=color, label='PnL')
        ax2.set_ylabel('PnL', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.yaxis.tick_left()  # 将PnL的Y轴移至左侧
        ax2.yaxis.set_label_position('left')  # 设置PnL标签位置到左侧

        # 仓位柱状图，现在将其设置为右侧Y轴
        color = 'tab:blue'
        ax1.bar(dates, positions, color=color, alpha=0.5)
        ax1.set_ylabel('Position', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(300))
        ax1.yaxis.tick_right()  # 将仓位的Y轴移至右侧
        ax1.yaxis.set_label_position('right')  # 设置仓位标签位置到右侧

        # 调整布局
        # fig.tight_layout()
        plt.title(title, fontproperties=zh_font1)
        plt.show()

    # 标的价格曲线 + 信号曲线
    def signal_curve(self, title):
        # 用spread充当信号
        df = self.price_df
        signal = df['spread']
        positions = self.alphas
        dates = df.index.tolist()

        fig, ax1 = plt.subplots(figsize=(15, 6))

        # 信号曲线
        color = 'tab:red'
        ax1.plot(dates, signal, color=color, label='Signal')
        ax1.set_ylabel('Signal', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(300))
        ax1.plot(df['mean'], label='Mean')

        # 仓位柱状图，现在将其设置为右侧Y轴
        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.bar(dates, positions, color=color, alpha=0.5)
        ax2.set_ylabel('Position', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(300))
        ax2.yaxis.tick_right()  # 将仓位的Y轴移至右侧
        ax2.yaxis.set_label_position('right')  # 设置仓位标签位置到右侧
        fig.legend()

        # fig.tight_layout()
        plt.title(title, fontproperties=zh_font1)
        plt.show()


def sensitivity_test(contracts, proportion):
    calmar_ratios = []
    strategy = CommodityFutureArbitrage(contracts, proportion)
    strategy.fetch_data()
    # 条带长度控制在0.2*sigma
    pairs = [[x, x+0.2] for x in np.arange(0, 2, 0.1)]
    for pair in pairs:
        strategy.calculate_signals(close=pair[0], distal=pair[1])
        strategy.trade()
        measures = measure(strategy.values, strategy.position)
        calmar_ratios.append(measures['calmar'])

    plt.plot(calmar_ratios)
    plt.show()


# 套利
def arbitrage(contracts, proportion, title):
    strategy = CommodityFutureArbitrage(contracts,  proportion)
    strategy.fetch_data()
    strategy.calculate_signals()
    strategy.trade()
    strategy.save_data(title)
    strategy.pnl(title)
    strategy.signal_curve(title)
    return strategy.values, strategy.position


class Measures:
    def __init__(self, values, position):
        self.values = values
        self.position = position

    def arr(self):
        end = self.values[-1]
        start = self.values[0]
        years = 1 / 252 * len(self.values)
        return (end / start) ** (1 / years) - 1

    def max_drawdown(self):
        values = self.values
        max_dd = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            dd = peak / value - 1
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def sharpe(self):
        arr = self.arr()
        std = np.std(self.values)
        return arr / std

    def calmar(self):
        arr = self.arr()
        max_dd = self.max_drawdown()
        return arr / max_dd

    # 此处为每日胜率，但调仓胜率更合理
    def win_rate(self):
        values = np.array(self.values)
        count = np.sum((values[1:] - values[:-1]) > 0)
        num = np.sum((values[1:] - values[:-1]) != 0)
        return count / num

    # 盈亏比
    def pcr(self):
        values = np.array(self.values)
        chg = values[1:] - values[:-1]
        profit = np.sum(chg[chg > 0]) / np.sum(chg > 0)
        coss = np.sum(chg[chg < 0]) / np.sum(chg < 0)
        return np.abs(profit / coss)

    # 平均持仓周期
    def avg_hold(self):
        filtered_position = self.position[(self.position != 0).all(axis=1)]
        total_period = filtered_position.shape[0]
        unique_position = filtered_position.drop_duplicates().shape[0]
        return total_period / unique_position


def measure(values, position):
    measures = Measures(values, position)
    dist = {
        'arr': measures.arr(),
        'max_drawdown': measures.max_drawdown(),
        'sharpe': measures.sharpe(),
        'calmar': measures.calmar(),
        'win_rate': measures.win_rate(),
        'pcr': measures.pcr(),
        'avg_hold': measures.avg_hold()
    }
    series = pd.Series(dist, index=dist.keys())
    return series


if __name__ == '__main__':
    # 钢厂产业链套利策略净值
    values1, position1 = arbitrage(['R.CN.SHF.rb.0004', 'R.CN.DCE.i.0004', 'R.CN.DCE.j.0004'], [10, -1.6, -0.5], '钢厂产业链套利')
    measure_data1 = measure(values1, position1)
    measure_data1.to_excel('results/钢厂产业链套利表现.xlsx')
    # 甲醇制 PP 利润套利
    values2, position2 = arbitrage(['R.CN.DCE.pp.0004', 'R.CN.CZC.MA.0004'], [2, -3], '甲醇制 PP 利润套利')
    measure_data2 = measure(values2, position2)
    measure_data2.to_excel('results/甲醇制 PP 利润套利表现.xlsx')
    # 炼焦套利
    values3, position3 = arbitrage(['R.CN.DCE.j.0004', 'R.CN.DCE.jm.0004'], [0.6, 1.4], '炼焦套利')
    measure_data3 = measure(values3, position3)
    measure_data3.to_excel('results/炼焦套利表现.xlsx')
    print(measure_data3['sharpe'])









