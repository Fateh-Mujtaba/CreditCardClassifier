import pandas as pd
#%%
path="E:/BS-CS-Vl/Machine Leaining Projects/Credit_card/credit_customers.csv"
data_df=pd.read_csv(path)

#%%

data_df = data_df.drop('other_payment_plans', axis=1)
data_df = data_df.drop('other_parties', axis=1)
data_df[['sex', 'marriage']] = data_df.personal_status.str.split(" ", expand = True)

#%% label encoder for the output

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_df['class'] = le.fit_transform(data_df['class'])
data_df['foreign_worker'] = le.fit_transform(data_df['foreign_worker'])
data_df['checking_status'] = le.fit_transform(data_df['checking_status'])
data_df['savings_status'] = le.fit_transform(data_df['savings_status'])
data_df['employment'] = le.fit_transform(data_df['employment'])

#%% one hot encoding
from sklearn.preprocessing import OneHotEncoder

columns_to_encode = ['checking_status','job','own_telephone', 'credit_history', 'purpose', 'savings_status', 
                     'employment', 'personal_status', 'property_magnitude','housing','sex','marriage' 
                     ]
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_columns = ohe.fit_transform(data_df[columns_to_encode])
encoded_df = pd.DataFrame(encoded_columns, columns=ohe.get_feature_names_out(columns_to_encode))
data_df = pd.concat([data_df, encoded_df], axis=1)
data_df = data_df.drop(columns_to_encode, axis=1)
#%%

from sklearn.preprocessing import MinMaxScaler
norm_cols=['duration','credit_amount','installment_commitment',
           'residence_since','age','existing_credits','num_dependents']
scaler = MinMaxScaler()
data_df[norm_cols] = scaler.fit_transform(data_df[norm_cols])
#%% seperate features and labels
X=data_df.drop('class',axis=1)
Y=data_df['class']
#%%
print(data_df.info())
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%%
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train)

#%%
from sklearn.metrics import accuracy_score
y_pred1 = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy:.2f}")
