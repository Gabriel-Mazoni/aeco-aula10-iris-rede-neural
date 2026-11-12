import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def relu(z):
    return np.maximum(z, 0)


def relu_derivada(z):
    return z > 0


def softmax(logits):
    exponenciais = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


def entropia_cruzada(alvos, probabilidades):
    return -(alvos * np.log(np.clip(probabilidades, 1e-15, 1))).sum(axis=1).mean()


class RedeNeural:
    def __init__(self, semente=42):
        rng = np.random.default_rng(semente)
        self.w1 = rng.normal(0, np.sqrt(2 / 4), (4, 8))
        self.b1 = rng.normal(0, 0.01, (1, 8))
        self.w2 = rng.normal(0, np.sqrt(2 / 8), (8, 3))
        self.b2 = rng.normal(0, 0.01, (1, 3))

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)
        probabilidades = softmax(a1 @ self.w2 + self.b2)
        return z1, a1, probabilidades

    def treinar(self, x, y, epocas=1000, taxa=0.1):
        for epoca in range(1, epocas + 1):
            z1, a1, probabilidades = self.forward(x)
            perda = entropia_cruzada(y, probabilidades)

            # Gradiente conjunto da softmax e da entropia cruzada média.
            dz2 = (probabilidades - y) / len(x)
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            dz1 = (dz2 @ self.w2.T) * relu_derivada(z1)
            dw1 = x.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)

            self.w1 -= taxa * dw1
            self.b1 -= taxa * db1
            self.w2 -= taxa * dw2
            self.b2 -= taxa * db2

            if epoca == 1 or epoca % 100 == 0 or epoca == epocas:
                print(f"Época {epoca:4d} | perda: {perda:.4f}")


def main():
    iris = load_iris()
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    padronizador = StandardScaler()
    x_treino = padronizador.fit_transform(x_treino)
    x_teste = padronizador.transform(x_teste)
    classes = np.eye(3)
    y_treino = classes[y_treino]
    y_teste = classes[y_teste]

    rede = RedeNeural()
    rede.treinar(x_treino, y_treino)
    _, _, probabilidades = rede.forward(x_teste)
    acertos = np.count_nonzero(probabilidades.argmax(axis=1) == y_teste.argmax(axis=1))
    print(f"Acurácia no teste: {acertos / len(y_teste):.2%} ({acertos}/{len(y_teste)})")


if __name__ == "__main__":
    main()
