import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
import sklearn.metrics as metrics
warnings.filterwarnings("ignore")
 
data=pd.read_csv("forestfires.csv")
data.head()

data=np.array(data)
X= data[1:,1:-1]
Y= data[1:,-1]
X= X.astype('int')
Y=Y.astype('int')
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=0)

logreg=LogisticRegression()

logreg.fit(X_train, Y_train)
Y_pred= logreg.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test,Y_pred))
print("Precision:", metrics.precision_score(Y_test,Y_pred))


check=[int(x) for x in "80 6 12".split(" ")]
final=[np.array(check)]
res=logreg.predict_proba(final)
print(res)

pickle.dump(logreg,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))