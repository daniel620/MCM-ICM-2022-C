import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
d1 = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/神经网络26w.xlsx')
d2 = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/规划模型24w.xlsx')
d3 = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/一般模型11w.xlsx')
d4 = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/一般模型14w.xlsx')
d5 = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/一般模型16w.xlsx')
date = d1['date'].values

i = 0
model_name = ['Improved DNN','Omptimization','General Model 1', 'General Model 2', 'General Model 3']
for d in [d1,d2,d3,d4,d5]:
    plt.plot(d['value'].rolling(20).mean(),label=model_name[i],linewidth=0.8)
    plt.legend()
    i+=1
plt.xticks([i for i in range(len(date))], date,rotation=45)
# plt.yticks(np.linspace(-3.5e12,-3.0e12,6))
# plt.xticks(trainDateList)
plt.ylabel('value of portfolio')
plt.xlabel('date')
plt.legend()
x_major_locator=plt.MultipleLocator(180)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.title('{}'.format(portfolio_value[-1]))
plt.savefig('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/2决策模型/data/final_data_再尝试/五个模型收益对比图.png',dpi=600, bbox_inches='tight')
plt.show()
