#%%
import pandas as pd
df = pd.read_parquet('../dados/dados_clones.parquet')
df.head()
# %%
# rename columns
df = df.rename(columns={
    'Massa(em kilos)': 'massa_quilos',
    'General Jedi encarregado': 'general_jedi_encarregado',
    'Estatura(cm)': 'estatura_cm',
    'Distância Ombro a ombro': 'distancia_ombro_ombro',
    'Tamanho do crânio': 'tamanho_cranio',
    'Tamanho dos pés': 'tamanho_pe',
    'Tempo de existência(em meses)': 'tempo_existencia_meses',
    'Status ': 'status',
})
df.head()
# %%
df['status_bool'] = df['status'] == 'Apto'
df.head()
# %%
## estatisitca descritiva

print(df.groupby('status')[['estatura_cm','massa_quilos']].mean())
print('\nnão parece ter relação; valores próximos')
# %%
## caracteristicas dos fisicas dos clones tem algum impacto?
# %%
df.groupby(['distancia_ombro_ombro'])['status_bool'].mean()
# %%
df.groupby(['tamanho_cranio'])['status_bool'].mean()
# %%
df.groupby(['tamanho_pe'])['status_bool'].mean()
# %%

df.groupby(['general_jedi_encarregado'])['status_bool'].mean()

# %%
features = [
    'massa_quilos',
    'estatura_cm',
    'distancia_ombro_ombro',
    'tamanho_cranio',
    'tamanho_pe'
]

cat_features = ['distancia_ombro_ombro', 
                'tamanho_cranio',
                'tamanho_pe']

X = df[features]

from feature_engine import encoding
onehot = encoding.OneHotEncoder(variables=cat_features)
X = onehot.fit_transform(X)
X

# %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, df['status_bool'])
# %%
