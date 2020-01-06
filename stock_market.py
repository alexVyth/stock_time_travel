''' Time Travel Problem '''

import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StockProcess:

    def __init__(self, size='small'):
        self.file = size + '.txt'
        self.day_dist = 360
        if size == 'small':
            self.N_limit = 1000
            self.intraday_par = 1.01
        else:
            self.N_limit = 1000000
            self.intraday_par = 1.01
        self.N = 0
        self.income = 1
        self.owned_stocks = {}
        self.portfolio = []
        self.balance = []
        self.dates = []
        try:
            os.remove('big.txt')
        except FileNotFoundError:
            pass
        try:
            os.remove('small.txt')
        except FileNotFoundError:
            pass

    def load_dataset(self):
        # Manage Working Dir
        project_dir = os.getcwd()
        stocks_dir = os.path.join(project_dir, 'Stocks')
        stocks = os.listdir(stocks_dir)
        # Load each CSV stock file
        stock_data = []
        for stock in sorted(stocks):
            stock_dir = os.path.join(stocks_dir, stock)
            try:
                stock_data.append(pd.read_csv(stock_dir))
                stock_data[-1]['Stock'] = stock[:-4]
                self.owned_stocks[stock[:-4]] = 0
            # Bypass empty CSVs
            except pd.errors.EmptyDataError:
                pass
            if len(stock_data) > 50:
                break
        # Concat DataFrames into one
        self.data = pd.concat(stock_data, axis=0, ignore_index=True)

    def preprocessing(self):
        # Create new Column with transaction limit
        self.data['Limit'] = self.data['Volume'] * 0.1
        # Delete unused Columns
        del self.data['Volume']
        del self.data['OpenInt']
        self.data.sort_values(by=['Date'])

    def buy(self, day, stock, amount, tr_type):   # sell-close -> Close
        tr_type_label = tr_type.split('-')[1].capitalize()
        open_price = self.data[(self.data.Date == day)
                               & (self.data.Stock == stock)][tr_type_label]
        open_price = open_price.iat[0]
        limit = self.data[(self.data.Date == day)
                          & (self.data.Stock == stock)]['Limit']
        limit = limit.iat[0]
        buy_total = open_price * amount
        if self.income >= buy_total and amount <= limit:
            self.income -= buy_total
            self.owned_stocks[stock] += amount
            self.N += 1
            out = open(self.file, "a")
            out.write(f'{day} {tr_type} {stock[:-3].upper()} {int(amount)}\n')
            out.close()
        else:
            sys.exit('Wrong transfer')

    def sell(self, day, stock, amount, tr_type):
        tr_type_label = tr_type.split('-')[1].capitalize()
        open_price = self.data[(self.data.Date == day)
                               & (self.data.Stock == stock)][tr_type_label]
        open_price = open_price.iat[0]
        limit = self.data[(self.data.Date == day)
                          & (self.data.Stock == stock)]['Limit']
        limit = limit.iat[0]
        sell_total = open_price * amount
        if self.owned_stocks[stock] >= amount and amount <= limit:
            self.income += sell_total
            self.owned_stocks[stock] -= amount
            self.N += 1
            out = open(self.file, "a")
            out.write(f'{day} {tr_type} {stock[:-3].upper()} {int(amount)}\n')
            out.close()
        else:
            sys.exit('Wrong Transfer')

    def intraday(self, day):
        # Filter Stocks that cannot be bought
        daily = self.daily[self.daily['Open'] <= self.income]
        # Filter Stocks that give profit
        daily = daily[daily['Open'] < daily['Close']]
        # Calculate daily profits
        daily['Profit'] = daily['Close'] - daily['Open']
        daily['Num'] = np.where(np.floor(self.income / daily['Open'])
                                < daily['Limit'] / 2,
                                np.floor(self.income / daily['Open']),
                                np.floor(daily['Limit'] / 2))
        daily['TotalProfit'] = daily['Num'] * daily['Profit']
        daily = daily[daily['TotalProfit'] + self.income
                      >= self.intraday_par * self.income]
        if daily.shape[0] >= 1:
            trans_info = [daily.iloc[0]['Date'],
                          daily.iloc[0]['Stock'], daily.iloc[0]['Num']]
            self.daily_transactions.append([*trans_info, 'buy-open'])
            self.daily_transactions.append([*trans_info, 'sell-close'])
            # self.buy(*trans_info, tr_type='buy-open')
            # self.sell(*trans_info, tr_type='sell-close')
            print(day, self.income, self.N)

    def intraday_reverse(self, day):
        # Filter Stocks that cannot be sold
        # daily = daily[daily['Close'] <= self.income]
        # Filter Stocks that give profit
        daily = self.daily[self.daily['Open'] < self.daily['Close']]
        # Calculate daily profits
        daily['Profit'] = daily['Close'] - daily['Open']
        daily['Num'] = np.where(np.floor(self.income / daily['Open'])
                                < daily['Limit'] / 2,
                                np.floor(self.income / daily['Open']),
                                np.floor(daily['Limit'] / 2))
        daily['TotalProfit'] = daily['Num'] * daily['Profit']
        daily = daily[daily['TotalProfit'] + self.income
                      >= self.intraday_par * self.income]
        if daily.shape[0] >= 1:
            trans_info = [daily.iloc[0]['Date'],
                          daily.iloc[0]['Stock'], daily.iloc[0]['Num']]
            self.daily_transactions.append([*trans_info, 'buy-open'])
            self.daily_transactions.append([*trans_info, 'sell-close'])
            print(day, self.income, self.N)

    def non_intraday(self, day):
        pass

    def ordered_transactions(self):
        priority1 = ['buy-open', 'sell-open']
        priority2 = ['buy-low', 'sell-high']
        priority3 = ['buy-close', 'sell-close']
        ordered_trans = []
        for trans in self.daily_transactions:
            if trans[3] in priority1:
                ordered_trans.append(trans)
        for trans in self.daily_transactions:
            if trans[3] in priority2:
                ordered_trans.append(trans)
        for trans in self.daily_transactions:
            if trans[3] in priority3:
                ordered_trans.append(trans)
        for trans in ordered_trans:
            if 'buy' in trans[3]:
                self.buy(*trans)
            else:
                self.sell(*trans)

    def stats(self, day):
        # Calculate Portfolio per day
        owned = []
        for key in self.owned_stocks:
            if self.owned_stocks[key] != 0:
                owned.append(key)
        df_owned = self.daily[self.data['Stock'].isin(owned)]
        owned_price = 0
        for index, row in df_owned.iterrows():
            x = row['Close'] * self.owned_stocks[row['Stock']]
            owned_price += x
        port = owned_price + self.income
        self.portfolio.append(port)
        self.dates.append(day)
        self.balance.append(self.income)

    def day_processing(self):
        self.intr_data = self.data[self.data['Close'] > self.data['Open']]
        day = min(self.intr_data['Date'])
        day = '1970-01-01'
        date_max = max(self.intr_data['Date'])
        date_max = '1990-03-01'
        while day <= date_max:
            print(day)
            self.intr_data = self.intr_data[self.intr_data['Date'] >= day]
            self.daily = self.intr_data[self.intr_data['Date'] == day]

            self.daily_transactions = []
            self.intraday(day)
            self.intraday_reverse(day)
            self.non_intraday(day)
            self.ordered_transactions()
            self.stats(day)

            day = datetime.strptime(day, '%Y-%m-%d')

            # Move to next day
            day = day + timedelta(1)
            day = str(day.date())

    def plot_diagrams(self):
        x = self.dates
        y1 = self.balance
        y2 = self.portfolio
        labels = ['Balance', 'Portfolio']
        fig, ax = plt.subplots()
        # ax.plot(x, y2)
        ax.fill_between(x, 0, y2)
        # ax.plot(x, y1)
        ax.fill_between(x, 0, y1)
        ax.set(ylabel=labels)
        ax.legend(labels)
        fig.savefig("diagrams.png")
        plt.show()

    def prepend_to_file(self):
        with open(self.file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(self.N).rstrip('\r\n') + '\n' + content)

    def calculate_sequences(self):
        sequence.load_dataset()
        sequence.preprocessing()
        sequence.day_processing()
        sequence.plot_diagrams()
        sequence.prepend_to_file()


sequence = StockProcess(size='small')
sequence.calculate_sequences()
