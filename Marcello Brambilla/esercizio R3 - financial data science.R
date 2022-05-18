# Utilizzando i pacchetti quantmod e fOptions scrivere un file in R che risolve i seguenti punti:
#   1. Scaricare i prezzi di chiusura giornalieri di una serie finanziaria.
#   2. Stimare la volatilità storica dai log-rendimenti su base annua.
#   3. Valutare un'opzione di tipo put europeo utilizzando la funzione CRRBinomialTreeOption con:
#      strike = 1.05 * S0     (S0 è il prezzo corrente del sottostante).
#   4. Analizzare il comportamento del prezzo all'aumentare del numero di periodi.

library("fOptions")
library("quantmod")

BinomialTreeEuOption = function(S0, CALL_or_PUT, N_periods) {
  Delta = 1/360
  sigma = 0.1 # sqrt(1/Delta*Sample2ndMom)  #logRet = diff(log(S0)) #Sample2ndMom = mean(logRet^2) # il momento secondo campionario del log rendimenti
  TimeToMat = 30/360 # un mese espresso come frazione dell'anno 
  if(CALL_or_PUT == 'CALL')
    call_or_put = 'ce'
  else
    call_or_put = 'pe'
  
  R = 0.01
  STRIKE = 1.05 * S0
  OPTION = CRRBinomialTreeOption(TypeFlag = call_or_put, S=S0, X=STRIKE, Time=TimeToMat, r=R, b=R, sigma=sigma, n=N_periods)
  return(OPTION@price)
}


aapl = getSymbols("AAPL")

v_close = as.numeric(AAPL$AAPL.Close)
v_returns = diff(log(v_close))
S0 = tail(v_close, 1L)

Sample2ndMom = mean(v_returns^2)# il momento secondo campionario del log rendimenti
delta_days = 360 # 1(1 day), 360 (1 year)
sigma = sqrt(delta_days * Sample2ndMom)


v_periods = c()
v_prices = c()
for (ST in 1:30) {
  PUT = BinomialTreeEuOption(S0, 'PUT', ST)
  v_prices[ST] = PUT
  v_periods[ST] = ST
}
plot(x=v_periods, y=v_prices, type="l")
  

