from core.predict_run import prediction_for_webpage
import pandas as pd
import datetime





#############################################################################################
## 테스트용 엑셀 로드
#df_all = pd.read_excel('data/han_2019.xlsx')
df_all = pd.read_excel('../web_data/han_2019-2020.xlsx')

print(df_all)

start_day = datetime.datetime(2019,6,1)
end_day = datetime.datetime(2019,6,15)
end_day += datetime.timedelta(days=1)

df = df_all.loc[(df_all[df_all.columns[0]]>=start_day) & (df_all[df_all.columns[0]]<end_day)]
print(df.shape)
#############################################################################################

prediction = prediction_for_webpage()

## dataframe = 7+5일 총 12일 데이터
## watershed = 0:한강, 1:낙동강, 2:금강, 3:영산강
## Target index =  0: 용존산소(DO), 1:총유기탄소(TOC) 2:총질소(TN) 3:총인(TP), 4:클로로필-a(Chl-a)

#for i in range(5):
nse, pbias, input, label, pred = prediction.run(dataframe=df, watershed=0, target=1)

print(nse)
print(pbias)
print(input)
print(label)
print(pred)
