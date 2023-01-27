#Logistic Regression Model 생성/학습

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# LogisticRegression 모델 생성/학습
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)

# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_lr.predict(X_test)
print(classification_report(y_test,pred))


#!--

from xgboost import XGBClassifier
# XGBClassifier 모델 생성/학습
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_xgb.predict(X_test)
print(classification_report(y_test,pred))


#!--

# XGBClassifier 모델의 feature_importances_를 이용하여 중요도 plot
"""
각 컬럼은 우리는 모르지만 결과에 영향을 미치는 정도가 있습니다다.
그 컬럼이 뭔지 알려주고 이를 통해서, feature engineering의 근간이 되는 것입니다.
"""

plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()

#!---

#모델 학습 결과 분석

from sklearn.metrics import plot_precision_recall_curve

# 두 모델의 Precision-Recall 커브를 한번에 그리기 (힌트: fig.gca()로 ax를 반환받아 사용)
fig = plt.figure()
ax = fig.gca()
plot_precision_recall_curve(model_lr, X_test, y_test, ax=ax)
plot_precision_recall_curve(model_xgb,X_test,y_test, ax=ax)


from sklearn.metrics import plot_roc_curve

# 두 모델의 ROC 커브를 한번에 그리기 (힌트: fig.gca()로 ax를 반환받아 사용)
fig = plt.figure()
ax = fig.gca()
plot_roc_curve(model_lr,X_test, y_test,ax=ax)
plot_roc_curve(model_xgb,X_test,y_test,ax=ax)
