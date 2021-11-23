from matplotlib.pyplot import legend
import numpy as np
from numpy.lib.histograms import _histogram_dispatcher


def oracle(guess=1, correct=6):
    if guess == correct:
        result = 'WIN'
    elif guess > correct:
        result = 'TOO HIGH'
    else:
        result = 'TOO LOW'
    return result


'''CODICE NUOVO'''

class cl_dummy_player(object): # cl_player eredita da object
  # ha variabili interne e metodi
  # i metodi con il doppio __ sono predefiniti, noi stiamo sovrascrivendo il metodo init
  def __init__(self, min_value, max_value, f_oracle): # questa __init__ è un metodo costruttore, che crea un oggetto. Viene chiamata quando inizializzo l'oggetto
    # self serve a fare riferimento ai suoi stessi dati
    # f_oracle: passo il risultato della funzione oracle
    self.mn = min_value
    self.mx = max_value
    self.f_oracle = f_oracle
    self.history = []

  def play(self, jackpot):
    trials = 0
    for guess in range(self.mn, self.mx):
      trials += 1
      outcome = self.f_oracle(guess = guess, correct = jackpot)
      if outcome == 'WIN':
          break
    self.history.append(trials)


class cl_smart_player(cl_dummy_player): # se non cambio nulla sarà uguale a cl_dummy_player
  def __init__(self, min_value, max_value, f_oracle):
    super(cl_smart_player, self).__init__(min_value, max_value, f_oracle)

  def play(self, jackpot): # sovrascrivo la funzione
    trials = 0
    initial_guess = (self.mx - self.mn) // 2
    for n in range(self.mn, self.mx):
      trials += 1
      outcome = oracle(guess=initial_guess, correct=jackpot)
      if outcome == 'WIN':
        break
      else:
        if outcome == 'TOO HIGH':
          initial_guess -= 1
        else:
          initial_guess += 1
    self.history.append(trials)


class cl_af_player(cl_dummy_player): # se non cambio nulla sarà uguale a cl_dummy_player
  def __init__(self, min_value, max_value, f_oracle):
    super(cl_af_player, self).__init__(min_value, max_value, f_oracle)

  def play(self, jackpot): # sovrascrivo la funzione
    trials = 0
    my_guess = last_guess = (self.mx - self.mn) // 2
    Max, Min = self.mx, self.mn
    #print('jackpot', jackpot, '   my_guess', my_guess)
    for n in range(self.mn, self.mx):
      trials += 1
      outcome = oracle(my_guess, jackpot)
      if outcome == 'WIN':
        break
      else:
        if outcome == 'TOO HIGH':
          Max = my_guess
          my_guess = (my_guess + Min) // 2
          if(my_guess == last_guess):
            my_guess -= 1
        else:
          Min = my_guess
          my_guess = (my_guess + Max) // 2
          if(my_guess == last_guess):
            my_guess += 1
      last_guess = my_guess
    self.history.append(trials)


MIN, MAX, PLAYS = 0, 10, 1000
dummy_player = cl_dummy_player(MIN, MAX, oracle) # oppure cl_player.__init__
smart_player = cl_smart_player(MIN, MAX, oracle)
af_player = cl_af_player(MIN, MAX, oracle)

for match in range(PLAYS):
  jackpot = np.random.randint(low=MIN, high=MAX)
  dummy_player.play(jackpot)
  smart_player.play(jackpot)
  af_player.play(jackpot)
print(smart_player.history[0:10])
print('dummy:', np.mean(dummy_player.history))
print('smart:', np.mean(smart_player.history))
print('af   :', np.mean(af_player.history))

#print(type(my_obj)) # <class '__main__.cl_player'> ==> perchè sto lavorando su '__main__'

'''
PROGRAMMAZIONE A OGGETTI
  - riutilizzabilità del codice
  - codice più manutenibile
  - mettere insieme dati e funzioni

Creo giocatore come oggetto

Per la prox volta, 
- trasformare algo in oggetto (fatto ora)
- nel tutorial, leggere capitolo 9, programmazione a oggeti

PROX LEZIONE
- sabato 20 alle 17:00 ? --> dom10:00
'''