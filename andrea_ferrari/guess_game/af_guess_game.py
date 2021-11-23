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

def strategy_1(plays, mx, mn):
  v_history = []
  for match in range(plays):
      jackpot = np.random.randint(low=mn, high=mx)
      counter = 0
      for n in range(mn, mx):
          counter += 1
          outcome = oracle(guess=n, correct=jackpot)
          if outcome == 'WIN':
              break
      v_history.append(counter)
  return v_history


def strategy_2(plays, mx, mn):
  v_history = []
  for match in range(plays):
      jackpot = np.random.randint(low=mn, high=mx)
      counter = 0
      initial_guess = (mx - mn) // 2
      for n in range(mn, mx):
          counter += 1
          outcome = oracle(guess=initial_guess, correct=jackpot)
          if outcome == 'WIN':
              break
          else:
              if outcome == 'TOO HIGH':
                  initial_guess -= 1
              else:
                  initial_guess += 1
      v_history.append(counter)
  return v_history

def strategy_3(plays, mx, mn):
  v_history = []
  for match in range(plays):
    jackpot = np.random.randint(low=mn, high=mx)
    counter = 0
    my_guess = last_guess = (mx - mn) // 2
    Max, Min = mx, mn
    #print('jackpot', jackpot, '   my_guess', my_guess)
    for n in range(mn, mx):
      counter += 1
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
      #print('    outcome:', outcome,'   my_guess', my_guess, '    Max', Max, '     Min', Min)
    v_history.append(counter)
  return v_history


class c_matches:
  def __init__(self, v_history_1, v_history_2, v_history_3, plays, max, min):
    self.plays = plays
    self.max = max
    self.min = min
    self.test_size = max - min

    self.mean_1 = round(np.mean(v_history_1),2) if len(v_history_1) > 0 else None
    self.mean_2 = round(np.mean(v_history_2),2) if len(v_history_2) > 0 else None
    self.mean_3 = round(np.mean(v_history_3),2) if len(v_history_3) > 0 else None


def do_matches(plays, max, min, v_strategies_to_play):
  v_history_1 = strategy_1(plays, max, min) if '1' in v_strategies_to_play else []
  v_history_2 = strategy_2(plays, max, min) if '2' in v_strategies_to_play else []
  v_history_3 = strategy_3(plays, max, min) if '3' in v_strategies_to_play else []
  cl_matches = c_matches(v_history_1, v_history_2, v_history_3, plays, max, min)
  return cl_matches


def do_multi_size_matches(plays, my_min, initial_max, step_max, final_max, v_strategies_to_play, plot_results = True, log_scale = False):

  v_c_matches = []
  v_max = range(initial_max, final_max+1, step_max)
  for i, my_max in enumerate(v_max):
    v_c_matches.append(do_matches(plays, my_max, my_min, v_strategies_to_play))
    x = v_c_matches[-1]
    print('  ', i+1,'/', len(v_max), '  |    PLAYS:', x.plays, '  |   MAX:', x.max, '  |  mean_1:', x.mean_1, '     mean_2:', x.mean_2, '     mean_3:', x.mean_3)

  if(plot_results == True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 8))
    v_colors = ['black', 'red', 'gold']
    v_linestyles = ['solid', 'solid', 'solid'] # "solid", "dashed", "dashdot", "dotted"

    v_patches = []
    if('1' in v_strategies_to_play):
      ax.plot(v_max, [x.mean_1 for x in v_c_matches], color = v_colors[0], linestyle = v_linestyles[0])
      my_patch_1 = mpatches.Patch(color=v_colors[0], linestyle=v_linestyles[0], label='v_means_1')
      v_patches.append(my_patch_1)

    if('2' in v_strategies_to_play):
      ax.plot(v_max, [x.mean_2 for x in v_c_matches], color = v_colors[1], linestyle = v_linestyles[1])
      my_patch_2 = mpatches.Patch(color=v_colors[1], linestyle=v_linestyles[1], label='v_means_2')
      v_patches.append(my_patch_2)

    if('3' in v_strategies_to_play):
      ax.plot(v_max, [x.mean_3 for x in v_c_matches], color = v_colors[2], linestyle = v_linestyles[2])
      my_patch_3 = mpatches.Patch(color=v_colors[2], linestyle=v_linestyles[2], label='v_means_3')
      v_patches.append(my_patch_3)


    plt.legend(handles = v_patches)
    
    # LOG SCALE
    if(log_scale == True):
      ax.set_yscale('log')
    plt.show()


MIN, MAX, PLAYS = 0, 1000, 10000
MIN, MAX, PLAYS = 0, 100, 1000
v_strategies_to_play = ['1','2', '3']
c_M = do_matches(PLAYS, MAX, MIN, v_strategies_to_play)
print(' MIN:', c_M.min, '  MAX:', c_M.max, '   PLAYS:', c_M.plays,'  |  mean_1:', c_M.mean_1, '     mean_2:', c_M.mean_2, '     mean_3:', c_M.mean_3)
# MIN: 0   MAX: 10    PLAYS: 1000      |  mean_1: 5.47      mean_2: 3.57      mean_3: 2.9
# MIN: 0   MAX: 10    PLAYS: 1000000   |  mean_1: None      mean_2: None      mean_3: 2.9
v_strategies_to_play = ['1', '2', '3']
do_multi_size_matches(PLAYS, MIN, initial_max=100, step_max=100, final_max=1000, v_strategies_to_play = v_strategies_to_play, plot_results = True, log_scale = False)
