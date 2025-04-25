#%%
import pandas as pd

df = pd.read_excel('../dados/dados_frutas.xlsx')
df

# %%
## Como aplicar o método do slide para descobrir a fruta?

filtro_redonda = df["Arredondada"] == 1
filtro_suculenta = df["Suculenta"] == 1
filtro_vermelha = df["Vermelha"] == 1
filtro_doce = df["Doce"] == 1

df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%
## Como podemos fazer a máquina aprender?

from sklearn import tree

features = ["Arredondada","Suculenta","Vermelha","Doce"]
target = "Fruta"

X = df[features]
y = df[target]

# %%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)


# %%
# visualizando arvore de decisão
import matplotlib.pyplot as plt
plt.figure(dpi=600)

tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=features,
               filled=True)

##
# %%
# features = ["Arredondada","Suculenta","Vermelha","Doce"]
arvore.predict([[1, 1, 1, 1]]) # cereja
arvore.predict([[1, 1, 0, 1]]) # pera
# %%

# empate -> retorna por ordem alfabética (50% maçã / cereja)
probas = arvore.predict_proba([[1, 1, 1, 1]]) # cereja
print(pd.Series(probas[0], index=arvore.classes_))
# %%
arvore.predict_proba([[1, 1, 0, 1]]) # pera
print(pd.Series(probas[0], index=arvore.classes_))
# %%
