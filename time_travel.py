''' Time Travel Problem '''

import os
from datetime import datetime, timedelta
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates


class TimeTravel:
    """ Algorithm for the Time Travel Problem"""

    def __init__(self, size='small'):
        # Initialize problem parameters/environment
        self.file = size + '.txt'
        if size == 'small':
            self.N_limit = 1000
            self.intraday_par = 1.15
            self.interday_par = 1.25
        else:
            self.N_limit = 1000000
            self.intraday_par = 1.000001
            self.interday_par = 1.000002
        self.N = 0
        self.income = 1
        self.owned_stocks = {}

        self.portfolio = []
        self.balance = []
        self.dates = []
        try:
            os.remove(self.file)
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
                # Create Dictionary with owned_stocks
                stock_data[-1]['Stock'] = stock[:-4]
                self.owned_stocks[stock[:-4]] = 0
            # Bypass empty CSVs
            except pd.errors.EmptyDataError:
                pass
        # Concat DataFrames into one
        self.data = pd.concat(stock_data, axis=0, ignore_index=True)

    def preprocessing(self):
        # Create new Column with transaction limit
        self.data['Limit'] = self.data['Volume'] * 0.1
        self.data['Limit'] = self.data['Limit'].apply(np.floor)
        self.data.Limit = self.data.Limit.astype('uint32')
        # Delete unused Columns
        del self.data['Volume']
        del self.data['OpenInt']
        # Manage Date Column
        self.data['Date'] = pd.to_datetime(self.data['Date'],
                                           format='%Y-%m-%d')
        # Create Year column for convinience
        self.data['Year'] = self.data['Date'].dt.year
        # Sort by date
        self.data = self.data.sort_values(by=['Date'])

    def buy(self, day, stock, amount, tr_type):
        """ The very-basic function 'buy' for checking/executing buying
        transactions"""

        # Find label of transaction (Open/Close/Low/High)
        tr_type_label = tr_type.split('-')[1].capitalize()
        # Find price of transaction for specific stock/day
        price = self.data[(self.data.Date == day)
                          & (self.data.Stock == stock)][tr_type_label]
        price = price.iat[0]
        # Find limit of transaction for checking
        limit = self.data[(self.data.Date == day)
                          & (self.data.Stock == stock)]['Limit']
        limit = limit.iat[0]
        buy_total = price * amount

        # Check if transaction is legal
        if self.income >= buy_total and amount <= limit:
            # Update Values
            self.income -= buy_total
            self.owned_stocks[stock] += amount
            self.N += 1
            # Save Sequence to file
            out = open(self.file, "a")
            out.write(f'{day.strftime("%Y-%m-%d")} {tr_type} {stock[:-3].upper()} {int(amount)}\n')
            out.close()
        else:
            sys.exit('Wrong Transfer')

    def sell(self, day, stock, amount, tr_type):
        """ The very-basic function 'sell' for checking/executing selling
        transactions"""

        # Find label of transaction (Open/Close/Low/High)
        tr_type_label = tr_type.split('-')[1].capitalize()
        # Find price of transaction for specific stock/day
        open_price = self.data[(self.data.Date == day)
                               & (self.data.Stock == stock)][tr_type_label]
        open_price = open_price.iat[0]
        # Find limit of transaction for checking
        limit = self.data[(self.data.Date == day)
                          & (self.data.Stock == stock)]['Limit']
        limit = limit.iat[0]
        sell_total = open_price * amount
        # Check if transaction is legal
        if self.owned_stocks[stock] >= amount and amount <= limit:
            # Update Values
            self.income += sell_total
            self.owned_stocks[stock] -= amount
            self.N += 1
            # Save Sequence to file
            out = open(self.file, "a")
            out.write(f'{day.strftime("%Y-%m-%d")} {tr_type} {stock[:-3].upper()} {int(amount)}\n')
            out.close()
        else:
            sys.exit('Wrong Transfer')

    def intraday(self, day):
        cash_limit = self.income - self.planned
        # Filter Stocks that cannot be bought
        daily = self.daily[self.daily['Open'] <= cash_limit]
        # Filter Stocks that give profit
        daily = daily[daily['Open'] < daily['Close']]
        # Calculate daily profits
        daily['Profit'] = daily['Close'] - daily['Open']

        # Find the possible amount of buying/selling
        daily['Amount'] = np.where(np.floor(cash_limit / daily['Open'])
                                   < daily['Limit'] / 2,
                                   np.floor(cash_limit / daily['Open']),
                                   np.floor(daily['Limit'] / 2))

        # Find the Profit and filter remaining data with intraday parameter
        daily['TotalProfit'] = daily['Amount'] * daily['Profit']
        daily = daily[daily['TotalProfit'] + self.portfolio[-1]
                      >= self.intraday_par * self.portfolio[-1]]

        # Save transaction information
        stock_list = [x[1] for x in self.daily_transactions]
        # check that transactions exist and do not overlap with
        # interday transactions
        if daily.shape[0] >= 1 and\
                not(any(daily.iloc[0]['Stock'] == x for x in stock_list)):
            trans_info = [daily.iloc[0]['Date'],
                          daily.iloc[0]['Stock'], daily.iloc[0]['Amount']]
            self.planned += trans_info[-1] * daily.iloc[0]['Open']
            self.daily_transactions.append([*trans_info, 'buy-open'])
            self.daily_transactions.append([*trans_info, 'sell-close'])
            self.today_stocks.append(trans_info[1])

    def intraday_reverse(self, day):
        # Create list with owned stocks
        stocks = self.owned_stocks.items()
        owned_stocks_list = list({k: v for (k, v) in stocks if v > 0}.keys())

        # Filter Stocks that cannot be sold
        daily = self.daily[self.daily['Stock'].isin(owned_stocks_list)]
        # Filter Stocks that give profit
        daily = daily[daily['Open'] > daily['Close']]
        # Calculate daily profits
        daily['Profit'] = daily['Open'] - daily['Close']

        # Find the possible amount of buying/selling
        amount = []
        for index, row in daily.iterrows():
            amount.append(int(np.where(self.owned_stocks[row['Stock']]
                                       < row['Limit'] / 2,
                                       self.owned_stocks[row['Stock']],
                                       np.floor(row['Limit'] / 2))))

        # Find the Profit and filter remaining data with intraday parameter
        daily['TotalProfit'] = amount * daily['Profit']
        daily["Num"] = np.array(amount)
        daily = daily[daily['TotalProfit'] + self.income
                      >= self.intraday_par * self.income]

        # Save transaction information
        stock_list = [x[1] for x in self.daily_transactions]
        # check that transactions exist and do not overlap with
        # interday transactions
        if daily.shape[0] >= 1 and\
                not(any(daily.iloc[0]['Stock'] == x for x in stock_list)):
            trans_info = [daily.iloc[0]['Date'],
                          daily.iloc[0]['Stock'], daily.iloc[0]['Num']]
            self.daily_transactions.append([*trans_info, 'sell-open'])
            self.daily_transactions.append([*trans_info, 'buy-close'])

    def yearly(self, day):
        # check the cash limit on current day
        cash_limit = self.income - self.planned
        found = False
        # find possible transactions on current day
        for trans in self.yearly_trans:
            stock, min_day, max_day, low, high, limit = trans
            if day == min_day and not found:
                # if cash is not enough delete transaction
                if low >= cash_limit or stock in self.today_stocks:
                    self.yearly_trans.remove(trans)
                else:
                    # save transaction info for buy transactions
                    amount = np.floor(cash_limit / low)
                    if amount >= limit:
                        amount = np.floor(limit)
                    if amount >= 1:
                        profit = amount * (high - low)
                        total_income = profit + self.portfolio[-1]
                        expect_income = self.interday_par * self.portfolio[-1]
                        if total_income > expect_income:
                            found = True
                            self.planned += low * amount
                            trans_info = [min_day, stock, amount, 'buy-low']
                            self.daily_transactions.append(trans_info)
                        else:
                            self.yearly_trans.remove(trans)
                    else:
                        self.yearly_trans.remove(trans)
            elif day == max_day:
                # save transaction info for save transactions
                amount = self.owned_stocks[stock]
                if amount >= limit:
                    amount = limit
                if amount >= 1:
                    amount = np.floor(amount)
                    trans_info = [max_day, stock, amount, 'sell-high']
                    self.daily_transactions.append(trans_info)

    def ordered_transactions(self,):
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
            print(trans)
            if 'buy' in trans[3]:
                self.buy(*trans)
            else:
                self.sell(*trans)
        if len(ordered_trans) > 0:
            print(self.income)

    def year_processing(self, year, day):
        # Clean Dataset
        self.data = self.data[self.data['Date'] >= day]

        # Find and save the best transactions of the year
        self.yearly_trans = []
        year_data = self.data[self.data['Year'] == year]
        for stock in self.owned_stocks.keys():
            stock_data = year_data[year_data['Stock'] == stock]
            if not stock_data.empty:
                # Find day with min stock price
                minim = min(stock_data['Low'])
                stock_low = stock_data[stock_data['Low'] == minim]
                stock_low = stock_low.iloc[-1]
                date_min = stock_low['Date']
                limit_min = stock_low['Limit']

                # Find day with max stock price
                maxim = max(stock_data['High'])
                stock_high = stock_data[stock_data['High'] == maxim]
                stock_high = stock_high.iloc[0]
                date_max = stock_high['Date']
                limit_max = stock_high['Limit']

                # If day of min < day of max save the transaction
                if date_min < date_max:
                    limit = min([limit_max, limit_min])
                    trans_info = [stock, date_min, date_max, minim, maxim,
                                  limit]
                    self.yearly_trans.append(trans_info)

    def day_processing(self):
        # Find dataset's min/max day
        day_min = min(self.data['Date'])
        day_max = max(self.data['Date'])
        day = day_min
        # Initialize Stats
        self.portfolio.append(self.income)
        self.balance.append(self.income)
        self.dates.append(day_min)

        # Iterate through all days
        while day <= day_max and self.N <= self.N_limit:

            # If first day of year make yearly actions
            if (day.day == 1 and day.month == 1) or\
                    day == day_min:
                year = day.year
                self.year_processing(year, day)

            # Initialize lists
            self.today_stocks = []
            self.daily_transactions = []
            self.planned = 0

            # Filter Dataset on current day
            self.daily = self.data[self.data['Date'] == day]

            # Call main processes
            self.yearly(day)
            self.intraday(day)
            self.intraday_reverse(day)
            self.ordered_transactions()

            # Save Stats
            if day != day_min:
                self.stats(day)

            # Move to next day
            day = day + timedelta(1)

    def stats(self, day):
        # Find owned stocks and amount
        owned = []
        for key in self.owned_stocks:
            if self.owned_stocks[key] != 0:
                owned.append(key)
        df_owned = self.daily[self.data['Stock'].isin(owned)]

        # Calculate Portfolio per day
        owned_worth = 0
        for index, row in df_owned.iterrows():
            x = row['Close'] * self.owned_stocks[row['Stock']]
            owned_worth += x
        port = owned_worth + self.income

        # Save balance, date, portfolio per day to lists
        self.balance.append(self.income)
        if port <= 10 * self.portfolio[-1]:
            # Consider big falls as Saturdays/Sundays
            self.portfolio.append(self.portfolio[-1])
        else:
            self.portfolio.append(port)
        self.dates.append(day)

    def plot_diagrams(self):
        # Initialize plot/data
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        x = self.dates
        y1 = self.balance
        y2 = self.portfolio
        labels = ['Portfolio', 'Balance']

        # Create Figure
        fig, ax = plt.subplots()
        ax.fill_between(x, 0, y2)
        ax.fill_between(x, 0, y1)

        # Ticks/Labels/Legend
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale("log")
        fig.autofmt_xdate()

        ax.set(ylabel='Dollars')
        ax.legend(labels, loc='upper left')

        # Show and Save Figure
        fig.savefig("diagrams.png")
        plt.show()

    def prepend_to_file(self):
        # Prepend the number of sequences N to file
        with open(self.file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(self.N).rstrip('\r\n') + '\n' + content)

    def calculate_sequences(self):
        # Execute the Process
        sequence.load_dataset()
        sequence.preprocessing()
        sequence.day_processing()
        sequence.prepend_to_file()
        sequence.plot_diagrams()


sequence = TimeTravel(size='big')
sequence.calculate_sequences()
