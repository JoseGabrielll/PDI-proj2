# -*- coding: utf-8 -*-
#!/usr/bin/python3


from scipy.fftpack import dct
from scipy.fftpack import idct
from numpy import *

import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import numpy
import wave
import math

def ReadWave():
	try:
		#Retorna a taxa de amostragem e um vetor com os dados
		Taxa_Amostragem, Dados = scipy.io.wavfile.read("MaisUmaSemana.wav")

	except IOError:
		print("Arquivo nao encontrado")

	else:
		print("Arquivo aberto")
		return Taxa_Amostragem, Dados

def WriteWave(Taxa,Dados, Nome):

	#Converte o array para um padrão gravável
	Dados = numpy.array(Dados, dtype=numpy.int16)
	
	scipy.io.wavfile.write(Nome, Taxa, Dados)

	print("Novo arquivo gravado")

def PlotWave(X_freq, Dc_valor):

	#Cria um vetor de zeros com o tamanho de X_freq
	Dc = numpy.zeros(shape=(len(X_freq)))

	#Preenche o array com o valor dc
	Dc.fill(Dc_valor)

	plt.figure("Dominio da Frequencia")
	plt.xlabel("Frequencia")
	plt.ylabel("Amplitude")
	plt.plot(Dc, color='black')
	plt.plot(X_freq, color='blue')
	plt.show()

def nCossenos(X_freq, n):
    
    '''
    #Modulo = numpy.absolute(X_freq)
    Media = numpy.mean(X_freq)
    print(Media)
    '''
    len_ = len(X_freq)

	#Cria um vetor de zeros com o tamanho de X_freq
    X_cos = numpy.zeros(shape=len_)

	#Cria um vetor com o modulo de X_freq
    Module = numpy.absolute(X_freq)

	#Cria um vetor ordenado por indice
    Sort = numpy.argsort(Module)

	#Percorre o X_freq e copia os cossenos mais importantes para o novo vetor
    for i in range(len_-1, (len_-1)-n, -1):
        pos = Sort[i]
        X_cos[pos] = X_freq[pos]
    
    '''
    #Modulo = numpy.absolute(X_cos)
    Media2 = numpy.mean(X_cos)
    print(Media2)
    
    constante = (2*Media2) - Media 
    print(constante)
   
    
    for j in range(len_):
        X_cos[j] = X_cos[j]*constante  
    '''
    
    return X_cos
    
def DCT(x):
    
    N = len(x)

	#Calcula o nivel Dc do sinal
    Nivel_Dc = math.sqrt(2.0/(N*1.0))*math.sqrt(1.0/2.0)*sum(x)
	#DCT da biblioteca scipy
    
    X_freq = dct(x, norm='ortho')  
    return X_freq, Nivel_Dc
    

    X_freq = []

    

    for k in range(N):
        Soma = 0

        Const = math.sqrt(2.0/(N*1.0))

        if(k == 0):
            Ck = math.sqrt(1.0/2.0)
        else:
            Ck = 1

        for n in range(N):

			#2pi(k/2N)n
            Freq = (2.0*(math.pi)*k*n)/(2.0*(N*1.0))

			#kpi/2N
            Fase = (k*(math.pi))/(2.0*(N*1.0))

			#Soma = x[n] * cos(Freq + Fase)
            Soma += x[n] * math.cos(Freq + Fase)

		#Valor = Const * Ck * Soma
        Valor = Const * Ck * Soma

        X_freq.append(Valor)

    return X_freq, Nivel_Dc

def IDCT(X):

	#IDCT da biblioteca scipy
    x_time = idct(X, norm='ortho')
    return x_time

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

def Compac_ExpIdeal(X_freq, C):
    
    Xt = roll(X_freq, C)

    return Xt

def Compac_Exp(X_freq, C):
    Xt = [0]*len(X_freq)    
    
    for k in range(1,len(X_freq)-1):       
        if((k*C) >= len(X_freq)):
            return Xt
        
        Xt[round(k*C)] = X_freq[k]
        
    return Xt

def main():

	#Vetores de teste
    '''
    #Teste do Slide
    Entrada_Tempo = [11.6156, 5.9285, 2.1515, 0.4693, -0.5441, 0.9595, 3.6881, 4.0156]
    Entrada_Frequencia = [10, 5, 8.5, 2, 1, 1.5, 0, 0.1]
   
    X,DC = DCT(Entrada_Frequencia)
    PlotWave(X,DC)
    print(DC)
    print(X)
    
      
    X = IDCT(Entrada_Frequencia)
    print(X)
    PlotWave(X)
    '''
    
    #DCT audio + nivel DC
    '''    
    
    X,DC = DCT(Dados)
    PlotWave(X,DC)
    
    x1 = numpy.zeros(shape=100)
    PlotWave(x1,DC)
    '''
    
    #Preservar os cossenos mais importantes
    '''
    Taxa,Dados = ReadWave()
    Xf,DC = DCT(Dados)
    Xcos = nCossenos(Xf,1000)
    PlotWave(Xcos,0)
    
    Xt = IDCT(Xcos)
    WriteWave(Taxa, Xt, "NovoAudio.wav")    
    '''
    
    
    #Compactador / Expansor
    '''
    Taxa, Dados = ReadWave()
    Xf,DC = DCT(Dados)
    XcompacIdeal = Compac_ExpIdeal(Xf, 1000)
    PlotWave(XcompacIdeal,0)
    XtIdeal = IDCT(XcompacIdeal)
    WriteWave(Taxa, XtIdeal, "CompacIdeal.wav")    
    
    Xcompac = Compac_Exp(Xf, 0.6)
    PlotWave(Xcompac,0)
    XtNormal = IDCT(Xcompac)
    WriteWave(Taxa, XtNormal, "CompacNormal.wav")
    '''
		
if __name__ == "__main__":
    main()
