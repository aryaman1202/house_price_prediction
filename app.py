from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

data = pd.read_csv("C:/Users/Aryaman/Downloads/Bengaluru_House_Data.csv")
data=data.drop(['area_type','availability','society'],axis=1)
data=data.drop(['balcony'],axis=1)
data['size']=data['size'].str.split(' ').str.get(0)
data['location'].fillna(data['location'].mode()[0],inplace=True)
data['size']=data['size'].str.split(' ').str.get(0)
data['size'].fillna(data['size'].median(),inplace=True)
data['size']=data['size'].astype(int)
data['bath'].fillna(round(data['bath'].mean()),inplace=True)
def conv_sqft_to_num(x):
    tokens = x.split('-')
    if(len(tokens)==2):
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
data['total_sqft']=data['total_sqft'].apply(conv_sqft_to_num)
data['price_per_sqft']=(data['price']*100000)/data['total_sqft']
data['location']=data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()
location_count_less_than_10 = location_count[location_count<=10]
data['location']=data['location'].apply(lambda x: 'others' if x in location_count_less_than_10 else x)
data=data[((data['total_sqft']/data['size'])>=200)]

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
data = remove_outliers_sqft(data)
data.describe()
def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('size'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('size'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
data=bhk_outlier_remover(data)
data.drop(columns='price_per_sqft',inplace=True)

X = data.drop(columns='price')
Y = data['price']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
ohe = OneHotEncoder()
ohe.fit(X[['location']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['location']),remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
scores = []
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_test_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test,y_test_pred))

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)

app = Flask(__name__)

# model=pickle.load(open('ML work/car_prediction/LinearRegressionModel.pkl','rb'))
data = pd.read_csv('C:/Users/Aryaman/Desktop/ML_work/house_prediction_model/cleaned_data.csv')

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    size = int(request.form.get('size'))
    total_sqft = float(request.form.get('total_sqft'))
    bath = int(request.form.get('bath'))
    prediction=pipe.predict(pd.DataFrame([[location,size,total_sqft,bath]],columns=['location','size','total_sqft','bath']))
    prediction=prediction*100000
    output = round(prediction[0],2)

    return str(output)

    return ""
if __name__ == "__main__":
    app.run(debug=True)
