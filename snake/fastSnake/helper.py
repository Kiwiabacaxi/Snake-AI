import matplotlib.pyplot as plt
from IPython import display

# mandar pronto

plt.ion()  # ion é para o plot conseguir se atualizar em tempo real


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    # fazer um modelo sem GUI PARA TREINAR MAIS RAPIDO PLMDDS
    plt.title("Treinando")
    plt.xlabel("Numero de geracoes")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)
