
from collections import defaultdict
from typing import DefaultDict, List, Dict
import numpy as np


class Hangman(object):

    def __init__(self,
                 list_of_words: List[str],
                 word_len: int = 6,
                 trials: int = 8):
        self._idx = {}
        self.word_len, self.trials = word_len, trials
        self._update_index(list_of_words)
        self.selected_word = self._select_word()
        self.mask = ['_'] * len(self.selected_word)

    def _update_index(self, list_of_words: List[str]):
        for word in list_of_words:
            try:
                self._idx[len(word)].append(word)
            except KeyError:
                self._idx[len(word)] = [word]

    def get_words(self, length: int):
        try:
            return self._idx[length]
        except KeyError:
            return []

    def add_words(self, list_of_words: List[str]):
        self._update_index(list_of_words)

    def print_status(self):
        print(self.mask)
        print('\nRemaining trials {}'.format(self.trials))

    def play(self, guess: str):
        self.trials -= 1
        for i, ch in enumerate(self.selected_word):
            if guess[i] == ch:
                self.mask[i] = ch
        return self.mask

    def _select_word(self):
        try:
            candidates = self._idx[self.word_len]
            return np.random.choice(candidates).lower()
        except KeyError:
            print(
                'Unavailable words of len {}'.format(
                    self.word_len))







class Player_AF:
  def __init__(self, c_hangman: Hangman, v_list_of_words: List[str]):
    #super(Player_AF, self).__init__(hangman, list_of_words)
    self.hangman = c_hangman
    self.mask = self.hangman.mask
    self.LEN = len(self.mask) # lunghezza della parola da indovinare

    d_char_index = defaultdict(lambda: 0)
    self.v_list_of_words = v_list_of_words
    for word in v_list_of_words:
      for char in word:
        d_char_index[char] += 1
    self.v_char_frequency = [c for c,f in sorted(list(d_char_index.items()), key=lambda x: -x[1])]

    self.n_initial_guesses = 2 # quanti tentativi farò con le lettere più frequenti, es. 'eroait'

    self.s_words_tried = set() # le parole che ho provato e che so che non sono esatte
    self.s_letters_pending = set() # le lettere che so che sono nella parola, ma che non ho ancora provato, es. 'aaaaaaaa'
    self.s_letters_tried = set() # le lettere che so che sono nella parola, ma che non ho ancora provato, es. 'aaaaaaaa'
    self.v_d_letters_tried = [{'v_letters_tried': set(), 'found': None} for x in range(self.LEN)] # per ogni elemento creo un vettore delle lettere provate
    '''
      [
        {
          'v_called': [],
          'found': None or 'a'
        },
      ]
    '''


  def get_v_compatible_words(self):
    V = set()
    X = [x for x in self.v_list_of_words if len(x) == self.LEN]
    Y = self.mask
    for x in X:
      to_append = True
      for q, ch in enumerate(x):
        if(Y[q] == '_'):
          continue
        if(Y[q] != ch):
          to_append = False
          break
      if(to_append == True):
        V.add(x)
    return V


  def get_guess(self, i):
    my_guess = ''
    LEN = len(self.mask)
    if i < self.n_initial_guesses:
      '''Le prime N volte (counter<2) chiamo gli N caratteri più frequenti'''
      my_guess = ''.join([str(x) for x in self.v_char_frequency[i*LEN:(i+1)*LEN]])
      self.s_letters_tried.add(my_guess[0])
    else:
      v_compatible_words = [x for x in self.get_v_compatible_words() if x not in self.s_words_tried]
      if(len(v_compatible_words) <= self.hangman.trials): # sono sicuro di vincere
        my_guess = v_compatible_words[0]
        self.s_words_tried.add(my_guess)
        print('     sono sicuro di vincere....my_guess:', my_guess, '  ', len(v_compatible_words), '/', self.hangman.trials)
      else:
        print('     NON sono sicuro di vincere....my_guess:', my_guess, '  ', len(v_compatible_words), '/', self.hangman.trials)        
        ''' (A) se so che qualche lettera c'è la provo estesa su tutta la parola, es. 'aaaaaaaaa'
            (B) sennò provo stringhe per similarità (complicato)'''
        if(len(self.s_letters_pending) > 0): # (A)
          my_guess = self.s_letters_pending.pop() * self.LEN # seleziona e rimuove
          self.s_letters_tried.add(my_guess[0])
        else: # (B)
          '''(C) se non ho ancora trovato 1 lettera
             (D) trovo le lettere più probabili, di fianco a lettere già trovate, o di fianco a lettere più probabili'''
          if(sum([1 for x in self.mask if x == '_']) == self.LEN):  # (C)
            my_guess = ''.join([str(x) for x in self.v_char_frequency[i*LEN:(i+1)*LEN]])
            self.s_letters_tried.add(my_guess[0])
          else:
            '''Funzione che ritorna per ogni posizione la lettera più probabile in base a ciò che ho accanto (realizzato o no), scorrendo da dx a sx
                se non ho nulla ritorna la lettera non provata.'''
            print('PRE  self.mask:', self.mask)
            my_guess = self.mask
            while sum([1 for x in my_guess if x == '_']) > 0:
              my_guess = get_advanced_guess(self.mask, self.get_most_likely_letter)# * self.LEN
            print('POST self.mask:', self.mask, '   my_guess', my_guess)
            #my_guess = v_compatible_words[0]
            #self.s_words_tried.add(my_guess)

    return my_guess



  def get_most_likely_letter(self, L, modality, position):
    assert modality in ['before', 'after'], "Error: modality must be in ['before', 'after']"
    v_all_words = self.v_list_of_words # [x for x in self.v_list_of_words if len(x) == self.LEN]
    d_letter_frequency = defaultdict(lambda: 0)

    for x in v_all_words:
      if(L not in x):
        continue
      for k,y in enumerate(x):
        if(modality == 'before'):
          if(k == 0):
            continue
        else:
          if(k >= len(x)-1):
            continue
        d_letter_frequency[y] += 1

    v_letter_frequency = [c for c,f in sorted(list(d_letter_frequency.items()), key=lambda x: -x[1])]
    v_letters_tried = self.v_d_letters_tried[position]['v_letters_tried']
    for x in v_letter_frequency:
      if(x not in v_letters_tried):
        self.v_d_letters_tried[position]['v_letters_tried'].add(x)
        return x


  def try_to_guess(self, guess):
    #guess = self.char_frequency[self.char_to_try] * self.hangman.word_len
    self.mask = self.hangman.play(guess)
    v_letters_missing = set()
    for i, ch in enumerate(self.mask):
      self.v_d_letters_tried[i]['v_letters_tried'].add(guess[i])
      if ch != '_':
        self.v_d_letters_tried[i]['found'] = ch
      else:
        if(guess[i] in self.hangman.selected_word):
          v_letters_missing.add(guess[i])
    #print('v_letters_missing:', v_letters_missing, '   self.s_letters_tried:', self.s_letters_tried)
    self.s_letters_pending = self.s_letters_pending.union({x for x in guess if x in self.hangman.selected_word if x not in self.s_letters_tried and x in v_letters_missing})


  def play(self):
    n_trials = self.hangman.trials
    for i in range(n_trials+1):
      guess = self.get_guess(i)
      print(guess,'   ',self.mask, self.hangman.trials, '   ', self.hangman.selected_word)
      self.try_to_guess(guess)

      if '_' not in self.mask:
        print('Win!', self.mask, self.hangman.trials)
        break
      elif self.hangman.trials < 0:
        print('Perso!', self.mask)
        break



def get_prior_next_value(my_guess, i, prior_or_next):
  assert prior_or_next in ['prior', 'next'], "Error, prior_or_next must be in ['prior', 'next']"
  if(prior_or_next == 'prior'):
    try:
      return my_guess[i-1]
    except:
      return None
  elif(prior_or_next == 'next'):
    try:
      return my_guess[i+1]
    except:
      return None


def get_advanced_guess(my_guess, GMLL):
  '''
  Casi, 
    i == 0 e ho il valore successivo
    i == last e ho il valore

  '''
  STR = []
  for p,Q in enumerate(my_guess):
    prior_mask = get_prior_next_value(my_guess, p, 'prior')
    next_mask = get_prior_next_value(my_guess, p, 'next')
    
    if(Q != '_'):
      continue
    
    if(prior_mask == None and next_mask == None):
      continue
    
    if(next_mask not in [None, '_']):
      letter = GMLL(next_mask, 'after', p)
      my_guess[p] = letter
      print('after...my_guess[p]...p:',p, my_guess[p])

    elif(prior_mask not in [None, '_']):
      letter = GMLL(prior_mask, 'before', p)
      my_guess[p] = letter
      print('prior...my_guess[p]...p:',p, my_guess[p])
  print('   get_advanced_guess......return my_guess:', my_guess)
  return my_guess




file_path = 'DATA/words/1000_parole_italiane_comuni.txt'
file_path = 'andrea_ferrari/list_words.txt' # https://www.mit.edu/~ecprice/wordlist.100000
with open(file_path, 'r') as fhandle:
    lines = fhandle.readlines()
words = [w.strip('\n') for w in lines]


h = Hangman(words, word_len=6, trials=8)
player = Player_AF(h, words)
player.play()


