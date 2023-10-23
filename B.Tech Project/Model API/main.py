import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

rd=pd.read_csv("EDA_PP_F_2.csv",encoding='latin1')
print(rd)

label_enc = LabelEncoder()
rd['type'] = label_enc.fit_transform(rd['type'])
X = rd['tweet_text']
y = rd['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1));
train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['tweet_text', 'type']);
X_train = train_os['tweet_text'].values
y_train = train_os['type'].values


cv = CountVectorizer()
X_train_cv =  cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = tf_transformer.transform(X_train_cv)
X_test_tf = tf_transformer.transform(X_test_cv)

print("Fitting the model...")
rf=RandomForestClassifier()
rf.fit(X_train_tf,y_train)
#other= misc and gender=sex
sentiments = ["age","not bullying","miscellaneous","racism","religion","gender"]
all_categories_names=np.array(sentiments)
x=rf.predict(cv.transform(["hi hello"]))
print(all_categories_names[x])
print("Done fitting!")
# Save your model


print("Saving the model...")
joblib.dump(rf, 'model.pkl')
joblib.dump(cv,'vectorizer.pkl')
joblib.dump(tf_transformer,'transformer.pkl')
print("Model dumped!")






