from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import line

d = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/0数据预处理/data/price_cleaned.xlsx')
g = d['usdpm'].values
b = d['value'].values
date = d['date'].values
def plot_gold(is_show=True):
    plt.plot(range(len(g)),g,linewidth=0.5,label='Gold Price')
    plt.xticks([i for i in range(len(date))], date,rotation=45)
    plt.ylabel('Price(U.S.dollars per troy ounce)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    x_major_locator=plt.MultipleLocator(180)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0)
    plt.ylim(0,2500)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # plt.grid()
    # plt.title('{}'.format(portfolio_value[-1]))
    plt.savefig('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/0数据预处理/data/原始数据时序图/gold时序图.png',dpi=600, bbox_inches='tight')

    if is_show==True:
        plt.show()    

    return 

def plot_bitcoin(is_show=True):
    plt.plot(range(len(b)),b,linewidth=0.5,label='Bitcoin Price')
    plt.xticks([i for i in range(len(date))], date,rotation=45)
    plt.ylabel('Price(U.S.dollars per bitcoin)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    x_major_locator=plt.MultipleLocator(180)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0)
    plt.ylim(0,70000)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # plt.title('{}'.format(portfolio_value[-1]))
    # plt.grid()

    plt.savefig('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/0数据预处理/data/原始数据时序图/bitcoin时序图.png',dpi=600, bbox_inches='tight')

    if is_show==True:
        plt.show()

    return

plot_gold()
plot_bitcoin()
