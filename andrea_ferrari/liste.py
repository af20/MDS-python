'''LISTE, gli elementi sono sovrascrivibili (mutable)'''
L = [1,2,3,4,5,6]
L.append(4)
a = L.index(4) # la prima volta che compare
a = L.index(4, 4) # la prima volta che compare, con 'start' il primo elemento a partire da
x = L.reverse() # non restituisce nessun valore, ma reverta la lista 'inplace'(senza crearne una copia) | L[::-1] fa invece uno slicing (creando una copia) della lista, prendendo tt gli elementi al contrario, ma la lista in sè non è cambiata

# list comprehension
L2 = [x**2 for x in L]

matrix = [
  [1,2,3,4],
  [5,6,7,8],
  [9,10,11,12]
]
M_k = [[x[i] for x in matrix] for i in range(len(matrix))] # matrice trasposta, scambio le righe con le colonne


# ENUMERATE
for i,x in enumerate(L2):
  pass#print(i, x)


v_n = [4, 3, 2, 0, 1, 2, 3] # per ogni n°, produrre la somma di quel n° più tutti i successivi a lui
v_n2 = [sum(v_n[i:]) for i,x in enumerate(v_n)]
#print(v_n2)

'''
TUPLE sono immutable
'''
T = tuple(v_n)

def calculate(a,b):
  s = a+b
  d = a-b
  p = a*b
  d = a/b
  return s, d, p, d # ==> ritorno una tupla


'''
SETS (insiemi) usarli quando devo raccogliere una collezione ordinata e univoca
        il controllo è quindi più veloce (if x in INSIEME)
'''
Z = {1,2,3,5,4,'z', 'a'}
print(Z)
A = {'a', 'b', 'c'}
B = {'a', 'b', 'c'}
A.add('x')
A.add('a') # non viene aggiunto perchè è un duplicato
c = A.difference(B) # .union, .intersection
#print(c)

'''
DIZIONARI, sono dei set di coppie chiave-valore
'''
D = {'name': 'Andrea', 'cognome': 'Bianchi'}
E = {'nome': 'Lucia', 'cognome': 'Rossi'}
for key, value in D.items(): # meglio di D.keys() e D.values()
  pass#print(key, value)

'''
 ESERCIZIO, prendo un testo, costriusco un indice che conta la frequenza dei caratteri (a,b,c,d,e)
    ==> voglio in output un dizionario
'''
import string
v_char = list(string.ascii_lowercase)

text = """Wikipedia è un'enciclopedia online, libera e collaborativa.
Grazie al contributo di volontari da tutto il mondo, 
Wikipedia è disponibile in oltre 300 lingue. Chiunque può contribuire alle voci esistenti o crearne di nuove, affrontando sia gli argomenti tipici delle enciclopedie tradizionali sia quelli presenti in almanacchi, dizionari geografici e pubblicazioni specialistiche.
"""

D = {}
for x in text.lower():
  if(x == '\n'):
    x = ' accapo'
  if x in D.keys():
    D[x] += 1
  else:
    D[x] = 1

sorted_dict = sorted(D.items(), key=lambda x: x[0])#, reverse=True)
#print(sorted_dict)


'''
FUNZIONI
  - Accettare un numero indefinito di parametri
  - chiamare una funzione all'interno di sè stessa
'''
def adder(*num):
    sum = 0
    for n in num:
        sum = sum + n
    print("Sum:",sum)


'''Accettare un numero indefinito di parametri'''
def arg_function(*args): # args (arguments), una lista
  print(list(args))

def kwargs_function(**kwargs): # kwargs (keyword arguments), una coppia chiave-valore
  print(kwargs)

#arg_function(1,2,3,4,1)
kwargs_function(**D)

'''
# chiamare una funzione all'interno di sè stessa
'''

'''
NOTE: nel loop l'operazione lenta è l'append, non il loop in sè. Nella list comprehension è più veloce perchè non fa append
      L'append è il doppio più lento circa
      i = 10000000 ==> loop: 4.74, list comprehension: 2.13
      i = 100000000 => memory error
'''
import time
i = 100000000

dt_st = time.time()
v = []
for x in range(i):
  pass#v.append(x)
print('loop:', round(time.time() - dt_st,2))

dt_st = time.time()
v = [x for x in range(i)]
print('list comprehension:', round(time.time() - dt_st,2))
