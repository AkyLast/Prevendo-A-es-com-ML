def warn(*args, **kwargs):
    pass
import warnings
warnings.warn
import pandas as pd

prev = pd.read_csv("C:/Users/Dia/Documents/Datasets/Ações_ML/Prev.csv")
print('Fechamento anterior:', prev["Close"][0])
print('Previsão anterior:', prev["Target"][0])


base = pd.read_csv("C:/Users/Dia/Documents/Datasets/Ações_ML/Hoje.csv")

try:
    amanha = pd.read_csv("C:/Users/Dia/Documents/Datasets/Ações_ML/Futuro.csv")
    print("Fechamento atual:", amanha["Close"][0])
    base = pd.concat([amanha, amanha[:1]], sort=True)
    amanha = amanha.drop(amanha[:1].index, axis=0)
    base.to_csv("C:/Users/Dia/Documents/Datasets/Ações_ML/Hoje.csv", index=False)
    amanha.to_csv(
        "C:/Users/Dia/Documents/Datasets/Ações_ML/Futuro.csv", index=False)
except Exception:
    print("O fechamento ainda não ocorreu!")
    pass

base["Target"] = base["Close"][1: len(base)].reset_index(drop=True)
prev = base[-1::].drop("Target", axis=1)
treino = base.drop(base[-1::].index, axis=0)

treino.loc[treino["Target"] > treino["Close"], "Target"] = 1
treino.loc[treino["Target"] != 1, "Target"] = 0

x = treino.drop("Target", axis=1)
y = treino["Target"]

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

print("Acurácia:", modelo.score(x_teste, y_teste))

prev["Target"] = modelo.predict(prev)
print("Fechamento de ontem:", prev["Close"][0])

if prev["Target"][0] == 1:
    print("VAI SUBIR!!")
else:
    print("Vai cair.")

prev.to_csv("C:/Users/Dia/Documents/Datasets/Ações_ML/Prev.csv", index=False)
