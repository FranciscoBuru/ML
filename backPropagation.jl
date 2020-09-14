using LinearAlgebra
using Random
using Plots

#Algoritmo de BackPropagation-------------------------------------------------

function backPropagation(in, out, f1, f1p, f2, f2p, W1, b1, W2, b2, alpha = 0.1)
    #Hacia Adelante
    n1 = W1*in + b1
    a1 = f1(n1)
    n2 = W2*a1 + b2
    a2 = f2(n2)
    e = out - a2
    #Hacia atras
    s2 = -2*f2p(n2)*e
    aux = f1p(n1)
    s1 = [aux[1] 0 ; 0 aux[2]]*transpose(W2)*s2
    #redefino pesos y bias
    W2r = W2 - alpha*s2*transpose(a1)
    b2r = b2 - alpha*s2
    W1r = W1 - alpha*s1*transpose(out)
    b1r = b1 - alpha*s1
    return W1r, b1r, W2r, b2r, e
end

#-----------------------------------------------------------------------------
#
#Valores Iniciales

W11 = [-0.27, -0.41]
b11 = [-0.48, -0.13]
W22 = [0.09, -0.17]
b22 = 0.48
W22 = transpose(W22)
valsInicio = [W11, b11, W22, b22]


#-----------------------------------------------------------------------------
#Evaluaciones-----------------------------------------------------------------

numero = 1000000
alpha1 = 0.01
inicio = -2
fin = 2
seed = 15


params1, params2 = dameParametros(numero,inicio,fin, seed)
err, resp , vals = evalua(params1, params2, valsInicio, alpha1)

#Resultado de evaluar en la red.
rectifica(1, vals[1], vals[2], vals[3], vals[4], logsig, purelin)

#Graficamos error
x = 1:numero
plot(x,err)

#Graficamos cambio en W1,1
plot(x,resp[1])

#Graficamos cambio en W1,2
plot(x,resp[2])

#Graficamos cambio en b1,1
plot(x,resp[3])

#Graficamos cambio en b2,2
plot(x,resp[4])


#Graficamos cambio en W2,1
plot(x,resp[5])

#Graficamos cambio en W2,2
plot(x,resp[6])

#Graficamos cambio en b2
plot(x,resp[7])



print("__\n")
print("__\n")


#-----------------------------------------------------------------------------
#Funciones útiles-------------------------------------------------------------

#Funcion para tomar num párametros en el intervalo [a,b]
function dameParametros(num, a, b, seed)
    Random.seed!(seed)
    ent = rand(num)*(b-a) + a*ones(num)
    z = zeros(num)
    for i in 1:num
        z[i] = 1+sin((pi*ent[i])/4)
    end
    return ent, z
end

#Funcion para rectificar/obtener la respuesta de la red neuronal
#al evaluar un valor
function rectifica(in, W1, b1, W2, b2, f1, f2)
        #Hacia Adelante
        n1 = W1*in + b1
        a1 = f1(n1)
        n2 = W2*a1 + b2
        a2 = f2(n2)
        print(a2)
        return a2
end

#Funcion para evaluar y entrenar.
function evalua(entradas, salidas, valores, alpha = 0.01)
    lenEnt = length(entradas)
    lenSal = length(salidas)
    error = zeros(lenSal)
    w11 = zeros(lenSal)
    w12 = zeros(lenSal)
    bb11 = zeros(lenSal)
    bb12 = zeros(lenSal)
    w21 = zeros(lenSal)
    w22 = zeros(lenSal)
    bb2 = zeros(lenSal)
    if lenSal == lenEnt
        for i in 1:lenSal
            valores = backPropagation(entradas[i], salidas[i],
                    logsig, logsigPrime, purelin, purelinPrime,
                    valores[1], valores[2], valores[3], valores[4], alpha)
            w11[i] = valores[1][1]
            w12[i] = valores[1][2]
            bb11[i] = valores[2][1]
            bb12[i] = valores[2][2]
            w21[i] = valores[3][1]
            w22[i] = valores[3][2]
            bb2[i] = valores[4]
            error[i]= valores[5]

        end
        arreRes = [w11, w12, bb11, bb12, w21, w22, bb2]
        return error, arreRes, valores
    end
end

#Funcion de capa del centro
function logsig(x)
    if length(x) == 1
        res = 1/(1+exp(-x))
    else
        res = zeros(length(x))
        for i in 1:length(x)
            res[i] = 1/(1+exp(-x[i]))
        end
    end
    return res
end

#Derivada de la funcion de la capa del centro
function logsigPrime(x)
    a = logsig(x)
    if length(x) == 1
        res = (1-a)*a
    else
        res = zeros(length(x))
        for i in 1:length(x)
            res[i] = (1-a[i])*a[i]
        end
    end
    return res
end

#Funcion de la ultima capa
function purelin(x)
    return x
end

#Derivada de la funcion de la ultima capa
function purelinPrime(x)
    return 1
end
