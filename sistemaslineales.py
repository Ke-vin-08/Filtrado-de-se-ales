"""
Una clase para operar con sistemas lineales

Utilización de las herramientas de sistemas ñineales

Nombre: Kevin David Ortega
Grupo: 1
Código: 1087210805
Fecha: 05/09/2022
"""

import numpy as np
import matplotlib.pyplot as plt
class SistemaLineal:
    """
    Clase para represntar un sistema lineal, causalm deterministico, invariante con el tiempo

    Métodos:
     - Definir Funcion de tranafesrencia
     - Definir modelo de espacio de estados
     - Respuesta forzada
        -- impulso
        -- escalon
    ## Metodos auxiliares
    -   ODE solver
    """
    perturbacion = []
    perturbacion0 = []
    act = []

    def __init__(self, tiposis='ft', *args):            #Documentación de una clase para crear el control
        """
        Inicialización de la ecuación diferencial
        :param tiposis: ’ft’: funcion de transferencia
        ’ee’: espacio de estados

        :param args: (num, den) para función de transferencia
        (A, B, C, D) para modelo de estados

        """
        # self.impi = impulso
        # self.vector = vetv

        self.tipo = tiposis

        self.dt = 0.1
        self.params = dict()
        if tiposis == 'ft':
            self.params.update({'num': np.array(args[0])})
            self.params.update({'den': np.array(args[1])})
            self.orden = len(args[0])
            self.ee(usuario=False)

        elif tiposis == 'ee':
            self.params.update({'A': np.array(args[0])})
            self.params.update({'B': np.array(args[1])})
            self.params.update({'C': np.array(args[2])})
            self.params.update({'D': np.array(args[3])})
            self.orden = self.params['A'].shape[0]
        else:
            raise NameError('El tipo de EDOL  no es válido')

    def ee(self, usuario=True):
        """
        Este método convierte la representación del sistema a
        un modelo de espacio de estados.
        :return: Nada (Se hace de forma interna en el objeto)
        """

        if self.tipo == 'ft':
            if usuario:
                self.tipo = 'ee'
            # Corresponde a la conversión de la forma Observable
            self.params.update({'A': np.diag(np.ones(self.orden - 1), k=1)})
            self.params['A'][:, 0] = - self.params['den']
            self.params.update({'B': self.params['num'].reshape((self.orden, 1))})
            self.params.update({'C': np.zeros(self.orden)})
            self.params['C'][0] = 1
            # TODO: Implementar la conversión de la forma Controlable

    def backward_euler(self, entrada, y0, dt):              #Método diferencia hacia atras
        """
        Este método calcula la salida del sistema ante la entrada ’entrada’
        usando el método de diferencias hacia atras de Euler:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por ’u’ y ’h’
        """

        ma = self.params['A']
        mb = self.params['B']
        mc = self.params['C']
        vec_x = y0
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            vec_x = np.linalg.solve(np.eye(self.orden) - ma * dt,
                                    vec_x + mb * entrada[i] * dt)
        # print('aaaaa',sal_y)
        return sal_y

    def forward_euler(self, entrada, y0, dt):           #Metodo diferencia hacia adelante
        """
        Este método calcula la salida del sistema ante la entrada ’entrada’
        usando el método de diferencias hacia adelante de Euler:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por ’u’ y ’h’
        """

        ma = self.params['A']
        mb = self.params['B']
        mc = self.params['C']
        vec_x = y0
        vetv = list()  # Salida
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            # sal_y[i] = mc @ vec_x  ###
            vetv.append(mc @ vec_x)
            vec_x = vec_x + (ma @ vec_x + mb * entrada[i]) * dt
        # print("aa0",sal_y)

        return vetv         #Modifiqué alguna cosa de aquí por dimensionalidad.

#Desde aquí empieza el código.
#Explicación: Se crea la señal con ruido
if __name__ == "__main__":
    """
    En la función main, se ejecutará el filtrado de una señal seno, además se explicará el 
    uso de los métodos.
    Devolverá las gráficas comparadas
    """

    #A continuación se añaden los parḿetros para la señal
    #Nota: en el documento, se evidencia la respuesta de la funcion de tranasferencia, y las variables de espacio de estado,
    #además, se hicieron ajustes para que la señal pasará, dado a que es un filtro pasa bajas y no pasaba las frecuencias altas.
    f = 1000
    R = 100
    C = 10e-3
    L = 10e-3
    T = 1 / f
    N = 5
    fs = 30 * f
    ts = 1 / fs
    w = 2 * np.pi * f
    vp = 120
#-------------------------------------------------------------------------------------------
    t = np.arange(0, N * T, ts)             #Arreglo de tiempos para la señal sin ruido
#---------------------------------------------------------------
    # Señal con ruido de la libreria np.
    ysen = vp * np.sin(w * t)
#Se le llama perturbación al ruido con una desviación del 10 %
    perturbacion = []

    # for d in range(len(ysin)):
    #     FT1 = SistemaLineal('ft', [0, 1], [0, 100])
    #     distor = np.random.normal(0, 0.1)
    #     self.perturbacion += [ysin[d] + distor]

    #  Ciclo para agregar ruido a la señal 100 veces:
    listaruido = []
    for b in range(100):
        for i in range(len(ysen)):
            distor = np.random.normal(0, 0.1, None)
            ruidonormal = ysen[i] + ysen[i]*distor
            listaruido.append(ruidonormal)
        perturbacion.append(listaruido)
#------------------------------------------------------------------------------------------------------------------------
    lista = []      #Por lo general, la señal ya es una lista, así que se sacara el proemdio de cada lista de la lista
    for i in range(len(ysen)):
        suma = 0
        for j in range(len(perturbacion)):
            suma += perturbacion[j][i]
        promedio = suma/len(perturbacion)
        lista.append(promedio)
#/////////////////////////////////////////////////////////En el documento está la función de transferencia.
    #Expresada en la frecuencia compleja
    FT1 = SistemaLineal('ft', [1 / (R * C)], [1 / (R * C)])   #Nota: Por error de dimensión, se expresó en valores en tiempo
    FT2 = SistemaLineal('ft', [0, 1 / (L * C)], [R / L, 1 / (L * C)])
    ysen1 = FT1.backward_euler(ysen, [0], ts)
    ysen2 = FT2.backward_euler(ysen, [[0], [0]], ts)
    y3 = FT1.backward_euler(lista, [0], ts)
    y4 = FT2.backward_euler(lista, [[0], [0]], ts)


    #--------------------------------------------------------------------
    #En esta sección se grafican las señales.
    plt.figure(1)
    plt.plot(t, ysen, label = 'Señal sin ruido')
    plt.plot(t, lista, label = 'Señal con ruido')
    plt.legend()
    plt.title("Comparación de señales")
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.grid()


    #Para el circuito RC
#-------------------------------------------------------------------------
    plt.figure(2)
    plt.plot(t, ysen1, label = 'Respuesta del circuito RC')
    plt.plot(t, y3, label = 'Señal filtrada para RC')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.title("Comparación de señales en RC")
    plt.grid()
#Para el circuit RLC
    #----------------------------------------------------------------------------------------------
    plt.figure(3)
    plt.plot(t, ysen2, label = 'Respuesta del circuito RLC')
    plt.plot(t, y4, label='Señal filtrada RLC')
    plt.legend()
    plt.title("Comparación de señales en RLC")
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()


#------------------------------------------------