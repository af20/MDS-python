# ES.1 Scrivete una funzione in R che valuta un'opzione europea di tipo put in un albero binomiale uniperiodale.
#   - Gli argomenti della funzione sono: S0 , K, t0 (valore di default 0), T (valore di default 1) e r (regime composto);
#   - la funzione restituisce una lista contenente 
#       - il prezzo della put a t0 , il vettore
#       - Q = (q u , q d ) 
#       - le quantità y* di titolo rischioso nel portafoglio di replica.

# ES.2 Scrivete una funzione in R che internamente richiama la funzione del punto 1 
#     e calcola il prezzo al tempo t 0 di una call europea con la put-call parity 
#     (ricordatevi di utilizzare il regime composto anche nella put-call parity per coerenza con il punto 1).

library("fOptions")
library("quantmod")
BinomialTreeEuOption = function(S0, STRIKE, R, CALL_or_PUT, T0=0, T1=1) {
  Delta = 1/360
  sigma = 0.1 # sqrt(1/Delta*Sample2ndMom)  #logRet = diff(log(S0)) #Sample2ndMom = mean(logRet^2) # il momento secondo campionario del log rendimenti
  TimeToMat = 30/360 # un mese espresso come frazione dell'anno 
  N_period = 1
  if(CALL_or_PUT == 'CALL')
    call_or_put = 'ce'
  else
    call_or_put = 'pe'
  
  OPTION = CRRBinomialTreeOption(TypeFlag = call_or_put, S=S0, X=STRIKE, Time=TimeToMat, r=R, b=R, sigma=sigma, n=N_period)
  return(OPTION@price)
}



# ES.2
PutCallParityPrice = function(S0, STRIKE, R, CALL_or_PUT, T0=0, T1=1) { # S: St oggi
  # C + PV(x) = P + S
  #   C = price of the European call option
  #   PV(x) = the present value of the strike price (x), discounted from the value on the expiration date at the risk-free rate
  #   P = price of the European put
  #   S = spot price or the current market value of the underlying asset
  TimeToMat = 30/360 # un mese espresso come frazione dell'anno 
  price_1 = BinomialTreeEuOption(S0, STRIKE, R, CALL_or_PUT)
  
  PV_K = STRIKE * (1+R)^TimeToMat

  if(CALL_or_PUT == 'CALL') {
    call_or_put = 'ce'
    price_parity = price_1 + PV_K - S0
  }
  else {
    call_or_put = 'pe'
    price_parity = price_1 + S0 - PV_K
  }
  
  return(price_parity)
}


# EX.2
mertonConstraints = function(PriceOpt, S, K, r, TypeCall=TRUE) { # S: St oggi
  TimeToMat = 30/360  
  if(TypeCall) {
    cond = PriceOpt >= max(S - K * exp(-r * TimeToMat), 0)
    cond = cond & S >= PriceOpt
  }else {
    cond = PriceOpt >= max(K * exp(-r * TimeToMat)-S, 0)
    cond = cond & K * exp(-r * TimeToMat)>=PriceOpt
  }
  return(cond)
}



S0 = 100
STRIKE = 101
R = 0.01 # tasso di interesse 
price = BinomialTreeEuOption(S0, STRIKE, R, 'PUT')
cat('price', price)
price_parity = PutCallParityPrice(S0, STRIKE, R, 'PUT')
cat('price_parity:', price_parity)
merton_constraints_PUT = mertonConstraints(price, S0, STRIKE, R, TypeCall=FALSE)
merton_constraints_CALL = mertonConstraints(price_parity, S0, STRIKE, R, TypeCall=TRUE)
cat('merton_constraints_PUT:', merton_constraints_PUT, '     merton_constraints_CALL:', merton_constraints_CALL)


