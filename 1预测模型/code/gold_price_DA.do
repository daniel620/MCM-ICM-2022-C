clear 
import excel "/Users/kuzaowuwei/Desktop/2022美赛/赛中数据/1预测模型/data/gold_price.xlsx", sheet("Sheet1") firstrow
gen t = _n
tset t
dfuller usdpm
ac usdpm
graph export 黄金价格ACF.png, replace 
pac usdpm
graph export 黄金价格PACF.png, replace 

// diff 1
dfuller d.usdpm
ac d.usdpm
graph export 黄金价格1阶差分ACF.png, replace 
pac d.usdpm
graph export 黄金价格1阶差分PACF.png, replace 

