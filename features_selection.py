import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

feat = pd.read_csv("allfeatures.csv")
X=feat.iloc[:,:26]
y=feat.iloc[:,-1]
corrmat=feat.corr()
tp_corr=corrmat.index
plt.figure(figsize=(40,40))
g=sns.heatmap(feat[tp_corr].corr(),annot=True)
g.get_figure().savefig('heatmap.png', bbox_inches='tight')



#from sklearn.ensemble import ExtraTreesClassifier
#model = ExtraTreesClassifier()
#le = preprocessing.LabelEncoder()
#y= le.fit_transform(feat['genre'])
#
#model.fit(X,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
##plot graph of feature importances for better visualization
#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
#
#feat[feat.columns[1:]].corr['genre'][:]

#import pandas as pd
#import numpy as np
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#y =preprocessing.LabelEncoder().fit_transform(feat.iloc[:,-1]) 
#feat['genre']=y
#bestfeatures = SelectKBest(score_func=chi2, k=6)
#
#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
##concat two dataframes for better visualization 
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(10,'Score'))  #print 10 best features
#
