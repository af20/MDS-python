
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

    def _select_word(self):
        try:
            candidates = self._idx[self.word_len]
            return np.random.choice(candidates).lower()
        except KeyError:
            print(
                'Unavailable words of len {}'.format(
                    self.word_len))




class Player_AF:
  def __init__(self, c_hangman: Hangman, v_list_of_words: List[str], use_advanced_guess):
    #super(Player_AF, self).__init__(hangman, list_of_words)
    self.hangman = c_hangman
    self.use_advanced_guess = use_advanced_guess
    d_char_index = defaultdict(lambda: 0)
    self.v_list_of_words = v_list_of_words
    for word in v_list_of_words:
      for char in word:
        d_char_index[char] += 1
    self.v_char_frequency = [c for c,f in sorted(list(d_char_index.items()), key=lambda x: -x[1])]


    self.n_initial_guesses = 2 # quanti tentativi farò con le lettere più frequenti, es. 'eroait'
    self.LEN = len(self.hangman.mask) # lunghezza della parola da indovinare

    self.s_words_tried = set() # le parole che ho provato e che so che non sono esatte
    self.s_letters_pending = [] # le lettere che so che sono nella parola, ma che non ho ancora provato, es. 'aaaaaaaa'
    self.s_letters_tried = set() # le lettere che so che sono nella parola, ma che non ho ancora provato, es. 'aaaaaaaa'
    self.v_d_letters_tried = [{'v_letters_tried': set(), 'found': None} for x in range(self.LEN)] # per ogni elemento creo un vettore delle lettere provate
    self.s_letters_absent = []



  def get_v_compatible_words(self):
    v_compatible_words = set()
    v_words = [x for x in self.v_list_of_words if len(x) == self.LEN]
    mask = self.hangman.mask
    for word in v_words: # x è una parola delle 100,000
      to_append = True
      for q, ch in enumerate(word):
        if(mask[q] == '_'): # skippo, per far rimanere to_append = True
          continue
        if(ch in self.s_letters_absent or mask[q] != ch): # SCARTO, se la parola contiene una lettera che so che non c'è, la scarto || se la maschera ha già una lettera diversa dalla lettera della parola in quella posizione
          to_append = False
          break
      if(to_append == True):
        v_compatible_words.add(word)
    
    v_compatible_words = [x for x in v_compatible_words if x not in self.s_words_tried]        
    return v_compatible_words


  def get_guess(self, i):
    my_guess = ''
    LEN = len(self.hangman.mask)
    if i < self.n_initial_guesses:
      '''Le prime N volte (counter<2) chiamo gli N caratteri più frequenti'''
      my_guess = ''.join([str(x) for x in self.v_char_frequency[i*LEN:(i+1)*LEN]])
      self.s_letters_tried.add(my_guess[0])
    else:
      v_compatible_words = self.get_v_compatible_words()
      if(len(v_compatible_words) <= self.hangman.trials): # sono sicuro di vincere
        my_guess = v_compatible_words[0]
        #print('     sono sicuro di vincere....my_guess:', my_guess, '  ', len(v_compatible_words), '/', self.hangman.trials)
      else:
        #print('     NON sono sicuro di vincere....my_guess:', my_guess, '  ', len(v_compatible_words), '/', self.hangman.trials)        
        ''' (A) se so che qualche lettera c'è la provo estesa su tutta la parola, es. 'aaaaaaaaa'
            (B) sennò provo stringhe per similarità (complicato)'''
        if(len(self.s_letters_pending) > 0): # (A)
          my_guess = self.s_letters_pending.pop() * self.LEN # seleziona e rimuove
          self.s_letters_tried.add(my_guess[0])
        else: # (B)
          '''(C) se non ho ancora trovato 1 lettera
             (D) trovo le lettere più probabili, di fianco a lettere già trovate, o di fianco a lettere più probabili'''
          if(sum([1 for x in self.hangman.mask if x == '_']) == self.LEN):  # (C)
            my_guess = ''.join([str(x) for x in self.v_char_frequency[i*LEN:(i+1)*LEN]])
            self.s_letters_tried.add(my_guess[0])
          else:
            '''Funzione che ritorna per ogni posizione la lettera più probabile, in base alla lettera che c'è accanto (già indovinata o stimata)'''
            if self.use_advanced_guess == True:
              my_guess = self.get_advanced_guess()
            else:
              my_guess = v_compatible_words[0]
    self.s_words_tried.add(my_guess)
    return my_guess

  def get_prior_next_value(self, i, d_my_guest, prior_or_next):
    assert prior_or_next in ['prior', 'next'], "Error, prior_or_next must be in ['prior', 'next']"
    if(prior_or_next == 'prior'):
      try:
        return d_my_guest[i-1]
      except:
        return None
    elif(prior_or_next == 'next'):
      try:
        return d_my_guest[i+1]
      except:
        return None



  def get_advanced_guess(self):
    d_my_guest = {i: x for i,x in enumerate(self.hangman.mask)}
    while sum([1 for q in d_my_guest.values() if q == '_']) > 0:
      for i,value in d_my_guest.items():
        prior_mask = self.get_prior_next_value(i, d_my_guest, 'prior')
        next_mask = self.get_prior_next_value(i, d_my_guest, 'next')
        
        if(value != '_'):
          continue

        if(prior_mask == None and next_mask == None):
          continue
        if(next_mask not in [None, '_']):
          d_my_guest[i] = self.get_most_likely_letter(next_mask, 'after', i)
          self.v_d_letters_tried

        elif(prior_mask not in [None, '_']):
          d_my_guest[i] = self.get_most_likely_letter(prior_mask, 'before', i)
    v_return = [x for x in d_my_guest.values()]
    str_return = ''.join([str(x) for x in v_return])
    return str_return



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
    self.hangman.play(guess)
    v_letters_missing = set()
    for i, ch in enumerate(self.hangman.mask):
      self.v_d_letters_tried[i]['v_letters_tried'].add(guess[i])
      if ch != '_':
        self.v_d_letters_tried[i]['found'] = ch
      else:
        if(guess[i] in self.hangman.selected_word):
          v_letters_missing.add(guess[i])
      if(guess[i] not in self.hangman.selected_word):
        self.s_letters_absent += guess[i]
    self.s_letters_pending += ({x for x in guess if x in self.hangman.selected_word if x not in self.s_letters_tried and x in v_letters_missing})
    self.s_letters_pending = list(set(self.s_letters_pending))
    self.s_letters_absent = list(set(self.s_letters_absent))




  def play(self, to_print = False):
    n_trials = self.hangman.trials
    for i in range(n_trials+1):
      guess = self.get_guess(i)
      if(to_print == True):
        print(guess,'   ',self.hangman.mask, self.hangman.trials, '   ', self.hangman.selected_word)
      self.try_to_guess(guess)

      if '_' not in self.hangman.mask:
        if(to_print == True):
          print('Win!', self.hangman.mask, self.hangman.trials)
        return 1
      if self.hangman.trials < 0:
        if(to_print == True):
          print('Perso!', self.hangman.mask)
        return 0




#file_path = 'DATA/words/1000_parole_italiane_comuni.txt'
file_path = 'andrea_ferrari/list_words.txt' # https://www.mit.edu/~ecprice/wordlist.100000
with open(file_path, 'r', encoding='utf-8') as fhandle:
    lines = fhandle.readlines()
words = [w.strip('\n') for w in lines]

multiple_play = True
use_advanced_guess = False

if(multiple_play == False):
  h = Hangman(words, word_len=6, trials=8)
  player = Player_AF(h, words, use_advanced_guess)
  player.play(to_print = True)
else:
  v_history = []
  for i in range(100):
    if(i%10 == 0):
      print('     ', i)
    h = Hangman(words, word_len=6, trials=8)
    player = Player_AF(h, words, use_advanced_guess)
    v_history.append(player.play())
  victory_rate = sum(v_history) / len(v_history)
  print(len(v_history), victory_rate)

# 100,000 parole | n_plays = 100 | word_len=6 | trials=8 
#  use_advanced_guess=False, ==> 86%, 80%, 84%, 80%
#  use_advanced_guess=True   ==> 65%, 64%


# 1,000 parole | word_len=6 | trials=8
#       use_advanced_guess=False, | n_plays = 100 ==> 99%  100%, 98% |  n_plays = 1000 ==> 99.7% 
#       use_advanced_guess=True,  | n_plays = 100 ==> 97%   98% |  n_plays = 1000 ==> 96.2%
