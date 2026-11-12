# Classificação de flores Iris com rede neural

Atividade da Aula 10 de Inteligência Artificial. A rede usa NumPy para implementar uma camada escondida com 8 neurônios ReLU e uma saída Softmax de 3 classes. O treinamento usa entropia cruzada, retropropagação e gradiente descendente em lote por 1.000 épocas, com taxa de aprendizado de 0,1.

O scikit-learn carrega o Iris, faz a divisão estratificada de 80% para treino e 20% para teste e ajusta a padronização usando o conjunto de treino. O script converte os rótulos em vetores one-hot, exibe a perda durante o treino e calcula a acurácia no conjunto de teste.

## Executar

```sh
python -m pip install -r requirements.txt
python iris_rede_neural.py
```
