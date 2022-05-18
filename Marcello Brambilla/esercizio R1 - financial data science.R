#EX.1

CallPayoff = function(ST, K, is_long=TRUE) {
  if(is_long)
    payoff = pmax(ST-K, 0)
  else
    payoff = K-ST
  return(payoff)
}



ButterflyPayoff = function(ST, K1, K2) {
 

  J = H = 10
  w = J / (J+H)
  K = w * K1 + (1 - w) * K2
  payoff = J * pmax(ST-K1,0) - (J+H) * pmax(ST-K, 0) + H*pmax(ST-K2, 0)
  return (payoff)
}


ButterflyChart = function(K1, K2, MIN=0, MAX=200) {
  v_payoff = c()
  v_strikes = c()
  for (ST in MIN:MAX){
    payoff = ButterflyPayoff(ST, K1, K2)
    v_payoff[ST] = payoff
    v_strikes[ST] = ST
  }
  #print('   MIN:MAX', length(MIN:MAX), '   v_payoff', length(v_payoff))
  plot(x=v_strikes, y=v_payoff, type="l")
}

# Payoff of a Call Option
CallPayoff(ST=100, K=20)

# Payoff of a Buttefly Strategy
ButterflyChart(K1=80,K2=90, MIN=50, MAX=120)
