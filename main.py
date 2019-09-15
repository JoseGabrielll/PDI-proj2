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

	len_ = len(X_freq)

	#Cria um vetor de zeros com o tamanho de X_freq
	X_filtrado = numpy.zeros(shape=len_)

	#Cria um vetor com o modulo de X_freq
	Module = numpy.absolute(X_freq)

	#Cria um vetor ordenado por indice
	Sort = numpy.argsort(Module)

	#Percorre o X_freq e copia os cossenos mais importantes para o novo vetor
	for i in range(len_-1, (len_-1)-n, -1):
		pos = Sort[i]
		X_filtrado[pos] = X_freq[pos]

	return X_filtrado
    
def DCT(x):

	#Calcula o nivel Dc do sinal
	Nivel_Dc = (1.0/2)*math.sqrt(1.0/2)*x[0]

	#DCT da biblioteca scipy
	#X_freq = dct(x, norm='ortho')
	#return X_freq, Nivel_Dc

	X_freq = []

	N = len(x)

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

	#IDCT da biblioteca scipy
	#x_time = idct(X, norm='ortho')
    #return x_time

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

def Compac_Exp(X_freq, C):

    Xt = roll(X_freq, C)

    return Xt
    
def main():

	#Vetores de teste
	Entrada_Tempo = [11.6156, 5.9285, 2.1515, 0.4693, -0.5441, 0.9595, 3.6881, 4.0156]
	Entrada_Frequencia = [10, 5, 8.5, 2, 1, 1.5, 0, 0.1]

		
if __name__ == "__main__":
    main()
