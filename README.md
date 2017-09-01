# Inteligencia Aritifical II
Este repositorio contiene los códigos fuentes de las practicas de Inteligencia Aritificial II.

## Practica 1. Perceptron
Se implementa un modelo de perceptron con 2 entradas (x,y) para problemas linealmente separables.
Contiene los modulos ia2.py y gui.py. el modulo ia2.py contiene la clase Perceptron2D

### Ejemplo
```
p = Perceptron2D()

p.xdata = [3,6,4,1]
p.ydata = [4,1,1,2]
p.output = [1,1,0,0]

p.train()

#plot
plt.plot([3,6],[4,1],'ro')
plt.plot([4,1],[1,2],'bo')
xdata = np.arange(-6,6,0.1)
plt.plot(xdata,p.recta(xdata))
plt.show()
```
