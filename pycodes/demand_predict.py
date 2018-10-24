import numpy as np

def unroll_coef(coef, number_goods, model = "QUAIDS"):
    if model == "QUAIDS" :
        nExogEq = number_goods + 3
        
    if model == "AIDS" :
        nExogEq = number_goods + 2
        
    alpha = coef[0]
    for i in np.arange(1, number_goods):
        alpha = np.vstack( (alpha, coef[0 + nExogEq * (i)] ) )
    alpha = alpha.ravel()
        
    beta = coef[1]
    for i in np.arange(1, number_goods):
        beta = np.vstack( (beta, coef[1 + nExogEq * (i)] ) )
    beta = beta.ravel()
    
    if model == "QUAIDS":
        lambd = coef[2]
        for i in np.arange(1, number_goods):
            lambd = np.vstack( ( lambd, coef[2 + nExogEq * (i) ] ) )
        lambd = lambd.ravel()
        
        gamma = coef[3:(3 + number_goods)]
        for i in np.arange(1, number_goods):
            gamma = np.vstack( (gamma, coef[(3 + nExogEq * (i)):(3 + 
                number_goods + nExogEq * (i ) ) ] ) )
        gamma = gamma.reshape( (number_goods, number_goods) )
        return(alpha, beta, gamma, lambd)
             
    if model == "AIDS":
        gamma = coef[2:(2 + number_goods)]
        for i in np.arange(1, number_goods):
            gamma = np.vstack( (gamma, coef[(2 + nExogEq * (i)):(2 + 
                number_goods + nExogEq * (i))] ) )
        gamma = gamma.reshape( (number_goods, number_goods) )
        return(alpha, beta, gamma)    
    
def price_index(coef, log_price, number_goods, alpha0 = 0, model = "QUAIDS"):
    if model == 'QUAIDS':
        alpha, beta, gamma, lambd = unroll_coef(coef, number_goods, model)
    if model == 'AIDS':
        alpha, beta, gamma = unroll_coef(coef, number_goods, model)
    
    lna = alpha0 + ( alpha * log_price.T ).sum()
    for i in np.arange(number_goods):
        for j in np.arange(number_goods):
            lna = lna + 0.5 * gamma[i,j] * log_price[i] * log_price[j]
    
    b = np.exp( (beta*log_price.T).sum() )
    return(lna,b)

def demand_predict(coef, log_price, log_expenditure, number_goods, alpha0 = 0, model = 'QUAIDS'):
    if model == 'QUAIDS':
        alpha, beta, gamma, lambd = unroll_coef(coef, number_goods, model)
    if model == 'AIDS':
        alpha, beta, gamma = unroll_coef(coef, number_goods, model)
    
    lna, b = price_index(coef, log_price, number_goods, alpha0, model)
    
    lexpadj = log_expenditure - lna
    if model == 'QUAIDS':
        lexpadjsq = lexpadj**2 / b
        
    pshare = np.zeros_like(log_price)
    if model == 'AIDS':
        for i in np.arange(number_goods):
            pshare[i]=alpha[i] + (gamma[i,]*log_price).sum() + beta[i] * lexpadj
    
    if model == 'QUAIDS':
        for i in np.arange(number_goods):
            pshare[i]= alpha[i] + (gamma[i,]*log_price).sum() + beta[i] * lexpadj + lambd[i] * lexpadjsq
            
    return(pshare)