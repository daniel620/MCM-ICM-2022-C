import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import tensorboard

def predict_price(asset):
    # gold or bitcoin
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe = pd.read_excel('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/data/price_cleaned.xlsx')
    dataset_gold = dataframe['usdpm'].values.reshape(-1,1)
    dataset_bitcoin = dataframe['value'].values.reshape(-1,1)
    dataset_date = dataframe['date'].values.reshape(-1,1)
    if asset == 'gold':
        dataset = dataset_gold.copy()
    else:
        dataset = dataset_bitcoin.copy()
    # print(dataset)
    # 将整型变为float
    dataset = dataset.astype('float32')
    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.70)
    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]
    trainDateList = dataset_date[:train_size]
    testDateList = dataset_date[train_size:]
    # 创建数据集
    def create_dataset(dataset, look_back):
    #这里的look_back与timestep相同
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return numpy.array(dataX),numpy.array(dataY)
    #训练数据太少 look_back并不能过大
    look_back = 1
    trainX,trainY  = create_dataset(trainlist,look_back)
    testX,testY = create_dataset(testlist,look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

    # make predictions
    if asset == 'gold':
        model = load_model("/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/code/DATA/LSTM_GOLD.h5")
    else:
        model = load_model("/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/code/DATA/LSTM_BITCOIN.h5")

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    print('loss function: mse')
    train_loss = model.evaluate(trainX,trainY,batch_size=1, verbose=2)
    test_loss=  model.evaluate(testX,testY,batch_size=1, verbose=2)
    print('train loss:\t', train_loss)
    print('test loss:\t', test_loss)


    #反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    trainDateList = trainDateList.reshape(-1,)
    testDateList = testDateList.reshape(-1,)
    # plt.subplot(1,2,1)
    w = 1
    # plt.plot(trainY,label='Real',linewidth=w)
    # plt.plot(trainPredict[1:],label='Predict',linewidth=w,linestyle='--')
    # plt.xticks([i for i in range(len(trainDateList))], trainDateList,rotation=45)
    # # plt.xticks(trainDateList)
    # plt.ylabel('price of '+asset)
    # plt.xlabel('date')
    # plt.legend()
    # plt.title('Training Set\n(MSE loss: {})'.format(train_loss))
    # # plt.title('trainLoss:'+str(train_loss))
    # x_major_locator=plt.MultipleLocator(180)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.subplot(1,2,2)
    plt.plot(testY,label='Real',linewidth=w)
    plt.plot(testPredict[1:],label='LSTM',linewidth=w,linestyle='--')
    plt.xticks([i for i in range(len(testDateList))], testDateList,rotation=45)

    # plt.xticks(trainDateList)
    y_label = ''
    if asset == 'gold':
        y_label = 'Price(U.S.dollars per troy ounce)'
    else:
        y_label = 'Price(U.S.dollars per bitcoin)'
    # Price(U.S.dollars per troy ounce)
    plt.ylabel(y_label)
    plt.xlabel('Date')
    plt.xlim(0)
    plt.legend()

    plt.title('Test Set\n(MSE loss: {:e})'.format(test_loss))
    x_major_locator=plt.MultipleLocator(180)
    ax=plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')

    ax.xaxis.set_major_locator(x_major_locator)
    # plt.title('testLoss:'+str(test_loss))
    plt.tight_layout()
    plt.savefig('/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/data/LSTM-Prediction/高清图/{}测试集时序图.png'.format(asset),dpi=600, bbox_inches='tight')

    plt.show()

predict_price('bitcoin')