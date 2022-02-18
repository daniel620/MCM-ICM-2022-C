clear
import delimited /Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/data/BCHAIN-MKPRU.csv
// sort date 
gen t = _n
tset t
// 对数收益率回归
// gen lnP = log(value)
// gen dLnP = lnP - l1.lnP
// reg  dLnP L.dLnP L2.dLnP L3.dLnP,r
// predict y 

// 单位根检验,lag可以没有
// dfuller value,lag(3)

//acf和pacf图
// ac value
// pac value
// ac d.value
// pac d.value

reg value L.value L1.value L2.value L3.value,r
predict pred_value
//q test
wntestq value,lag(3)

// 价格回归1
reg value L.value L1.value,r
predict pred_value
//q test
wntestq value,lag(1)


// outreg2 using AR_result_3back.doc
