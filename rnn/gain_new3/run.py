from predict_run import prediction_for_webpage
import pandas as pd
import datetime

prediction = prediction_for_webpage()


df_all = pd.read_excel('save_web/han_2019.xlsx')

print(df_all)

start_day = datetime.datetime(2019,6,1)
end_day = datetime.datetime(2019,6,12)
end_day += datetime.timedelta(days=1)

df = df_all.loc[(df_all[df_all.columns[0]]>=start_day) & (df_all[df_all.columns[0]]<end_day)]


print(df.shape)

for i in range(5):
    _, _, label, prdi = prediction.run(dataframe=df, watershed=0, target=i)
