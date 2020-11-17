import pandas as pd

data = pd.read_csv(r'LBW_Dataset.csv')

def pre_process(data):
		replace_mean = ['Age','Weight','HB','BP']
		for col in replace_mean:
			data[col].fillna(data[col].mean(),inplace = True)

		# Delivery Phase,IFA,Residence with forward fill 
		forward_fill = ['Delivery phase','IFA' ,'Residence']
		for col in forward_fill:
			data[col].ffill(axis = 0,inplace=True)

		# Community,Education with mode
		replace_mode = ['Community','Education','Result']
		mode=[data.mode()[col][0] for col in replace_mode]
		for i in range(len(replace_mode)):
			data[replace_mode[i]].fillna(mode[i],inplace = True)

		# Min max normalization
		cols=list(data.columns)
		cols.remove('Education')
		for column in cols:
			max_val = max(data[column])
			min_val = min(data[column])
			data[column] = (data[column]-min_val)/(max_val-min_val)
		data['Education']=1.0	

		return data	

data = pre_process(data)
data.to_csv('Cleaned_LBW_Dataset.csv',index = False)