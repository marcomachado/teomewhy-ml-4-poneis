#%%
import pandas as pd
df = pd.read_excel('../dados/dados_cerveja.xlsx')
df.head()
# %%
# descarta ID
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

X = df[features]
y = df[target]
# %%
X = X.replace({
    "mud":1, "pint":0,
    "sim":1, "não":0,
    "clara":1, "escura":0,
})
X
# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=600)
tree.plot_tree(arvore, feature_names=features, 
               class_names=arvore.classes_,
               filled=True)
# %%
proba = arvore.predict_proba([[-5, 0, 1, 0]])
pd.Series(proba[0], index=arvore.classes_, name='Probabilidade')
# %%
