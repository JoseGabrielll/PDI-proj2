# -*- coding: utf-8 -*-
#!/usr/bin/python3


from scipy.fftpack import dct
from scipy.fftpack import idct
import matplotlib.pyplot as plt
#import scipy.io.wavfile
import numpy as np
import numpy
import math

def OpenWave():
    Taxa_Amostragem, Dados = scipy.io.wavfile.read("MaisUmaSemana.wav")
    
    return Taxa_Amostragem, Dados

def Plot(X_freq, Dc_valor):

    Dc = []

    for i in range(len(X_freq)):
        Dc.append(Dc_valor)

    plt.stem(X_freq)
    plt.stem(Dc)
    plt.show()

def Filtro(X_freq, n):
    X_filtrado = []
    len_ = len(X_freq)

    #Cria um array de 0 com N(amostras) elementos
    for i in range(len_):
        X_filtrado.append(0)

    #Cria um array ordenado por valor na posicao
    Sort = numpy.argsort(X_freq)

    #Varre os n elementos selecionados e adiciona ao novo sinal filtrado
    for i in range(len_-1, (len_-1)-n, -1):
        pos = Sort[i]
        X_filtrado[pos] = X_freq[pos]
    
    return X_filtrado
    
def DCT(x):
    X_freq = []
    
    N = len(x)

    Nivel_Dc = (1.0/2)*math.sqrt(1.0/2)*x[0]
    
    for k in range(N):
        Soma = 0
        
        Const = math.sqrt(2.0/N)
        
        if(k == 0):
            Ck = math.sqrt(1.0/2)
        else:
            Ck = 1
        
        for n in range(N):

            #2pi(k/2N)n
            Freq = (2*(math.pi)*k*n)/(2*N)

            #kpi/2N
            Fase = (k*(math.pi))/(2*N)

            #Soma = x[n] * cos(Freq + Fase)
            Soma += x[n] * math.cos(Freq + Fase)

        #Valor = Const * Ck * Soma
        Valor = Const * Ck * Soma
        
        X_freq.append(Valor)
    
    return X_freq, Nivel_Dc

def IDCT(X):
    x_time = []

    N = len(X)

    for n in range(N):
        Soma = 0
        
        Const = math.sqrt(2.0/N)
        
        for k in range(N):
            if(k == 0):
                Ck = math.sqrt(1.0/2)
            else:
                Ck = 1

            #2pi(k/2N)n
            Freq = (2*(math.pi)*k*n)/(2*N)

            #kpi/2N
            Fase = (k*(math.pi))/(2*N)

            #Soma = Ck * X[k] * cos(Freq + Fase)
            Soma += Ck * X[k] * math.cos(Freq + Fase)

        #Valor = Const *Somatorio
        Valor = Const * Soma
        
        x_time.append(Valor)
    
    return x_time
    
def main():
    Entrada_Tempo = [11.6156, 5.9285, 2.1515, 0.4693, -0.5441, 0.9595, 3.6881, 4.0156]
    Entrada_Frequencia = [10, 5, 8.5, 2, 1, 1.5, 0, 0.1]

    #Teste DCT
    '''
    print("Entrada:",Entrada_Tempo)
    print("\n")
    
    print("Saida Esperada:",Entrada_Frequencia)
    print("\n")
    
    Result = dct(Entrada_Tempo, norm='ortho')
    print("DCT lib:     ",Result)
    print("\n")
    
    Result, Dc = DCT(Entrada_Tempo)
    print("DCT Humilde: ",Result)
    print("\n")

    Plot(Result, Dc)
    '''

    #Teste IDCT
    '''
    print("Entrada:",Entrada_Frequencia)
    print("\n")
    
    print("Saida Esperada:",Entrada_Tempo)
    print("\n")
          
    Result = idct(Entrada_Frequencia, norm='ortho')
    print("IDCT lib:     ",Result)
    print("\n")
    
    Result = IDCT(Entrada_Frequencia)
    print("IDCT Humilde: ",Result)
    print("\n")

    Plot(Result, 0)
    '''

    #Teste Filtro
    '''
    Result = Filtro(Entrada_Frequencia, 4)
    print("Cos mais importantes: ",Result)
    print("\n")

    Plot(Result, 0)
    '''
    
if __name__ == "__main__":
    main()
