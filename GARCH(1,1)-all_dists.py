import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln, digamma, polygamma, betaln

def distini(dist):
    if dist == 'normal':
        return []
    elif dist == 't':
        return [6]
    elif dist == 'GED':
        return [2]
    elif dist == 'EGB2':
        return [3, 3]


def distbounds(dist):
    if dist == 'normal':
        return []
    elif dist == 't':
        return [(2.1, 30)]    
    elif dist == 'GED':
        return [(0.1, 100)]
    elif dist == 'EGB2':
        return [(0.1, 100), (0.1, 100)]


#logpdf_array is an array with the loglikelihood value of every return observation given the conditional variance
#to get the total loglikelihood, this array is summed in the LLGARCH function
def logpdf_normal(theta, returns, cond_variance):
    logpdf_array = -0.5 * (np.log(2*np.pi) + np.log(cond_variance) + (returns**2) / cond_variance)
    return logpdf_array


def logpdf_t(theta, returns, cond_variance):
    Nu = theta[3]
    logpdf_array = gammaln((Nu + 1) / 2) - gammaln(Nu / 2) \
            - 0.5 * np.log((Nu - 2) * np.pi * cond_variance) \
            - 0.5 * (Nu + 1) * np.log(1 + (returns ** 2) / ((Nu - 2) * cond_variance))
    return logpdf_array


def logpdf_GED(theta, returns, cond_variance):
    Shape = theta[3]
    lambdaFunc = (gamma(1/Shape) / ((2 ** (2/Shape)) * gamma(3/Shape)))**0.5
    vLogPdfPart1 = np.log((2**(1+(1/Shape))) * gamma(1/Shape) * lambdaFunc)
    vLogPdfPart2 = 0.5 * np.log(cond_variance)
    vLogPdfPart3 = np.log(Shape) 
    vLogPdfPart4 = 0.5 * (np.abs((returns - 0)/(lambdaFunc * (cond_variance **0.5))) **Shape)
    logpdf_array = - vLogPdfPart1 - vLogPdfPart2 + vLogPdfPart3 - vLogPdfPart4
    return logpdf_array


def logpdf_EGB2(theta, returns, cond_variance):
    dP = theta[3]
    dQ = theta[4]

    delta = digamma(dP)-digamma(dQ)
    omega = polygamma(1, dP) + polygamma(1,dQ)

    repeat = np.sqrt(omega)*((returns)/np.sqrt(cond_variance)) + delta
    logpdf_array = 0.5 * np.log(omega) + dP * (repeat) -0.5 * np.log(cond_variance) \
    - betaln(dP, dQ) - (dP+dQ) * np.log(1+ np.exp(repeat))
    return logpdf_array

    

def logpdf(theta, dist, returns, cond_variance):
    if dist == 'normal':
        return logpdf_normal(theta, returns, cond_variance)
    if dist == 't':
        return logpdf_t(theta, returns, cond_variance)
    if dist == 'GED':
        return logpdf_GED(theta, returns, cond_variance)
    if dist == 'EGB2':
        return logpdf_EGB2(theta, returns, cond_variance)
    

#you can keep the calculation of the cond variance in the LLGARCH function, but creating it's own function for this
#may be usefull for plotting (and other uses in the future)
def calc_cond_variance(theta, dist, returns):
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]

    T = len(returns)
    cond_variance = np.zeros(T)
    cond_variance[0] = np.var(returns)

    for t in range(1, T):
        cond_variance[t] = omega + alpha * returns[t-1] **2 + beta * cond_variance[t-1]
        
    return cond_variance


def LLGARCH(theta, dist, returns):
    cond_variance = calc_cond_variance(theta, dist, returns)
    loglikelihood = np.sum(logpdf(theta, dist, returns, cond_variance))
    
    return -loglikelihood #maximizing llikelihood is minimizing negative llikelihood


def constraint(theta, dist, returns):
    #alpha + beta < 1 for stationarity, the fmin_slsqp function keeps the value in this array > 0, so alpha + beta < 1
    alpha = theta[1]
    beta = theta[2]

    return(np.array([1-alpha-beta]))


def estimateGARCH(dist, returns):
    omega_ini = 0.05
    alpha_ini = 0.05
    beta_ini = 0.9

    omega_bnds = (0, 1)
    alpha_bnds = (0, 1)
    beta_bnds = (0, 1)

    theta_ini = [omega_ini, alpha_ini, beta_ini] + distini(dist)
    theta_bnds = [omega_bnds, alpha_bnds, beta_bnds] + distbounds(dist)

    theta_ML = fmin_slsqp(LLGARCH, theta_ini, bounds=theta_bnds, args=(dist, returns,), f_ieqcons=constraint)

    return theta_ML


def GARCH(dist, returns):
    theta_ML = estimateGARCH(dist, returns)
    with np.printoptions(precision=3, suppress=True):
        print("ThetaML: ", theta_ML)
    cond_variance = calc_cond_variance(theta_ML, dist, returns)
    
    plt.plot(returns, label='returns')
    plt.plot(np.sqrt(cond_variance), label='volatility')
    plt.title('returns and volatility from GARCH(1, 1)-{} model'.format(dist))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    SP500data = pd.read_csv('SP500_1998-2007.csv')['Close']

    length_data = len(SP500data)
    # logreturns in %
    SP500returns = 100 * (np.array(np.log(SP500data[1:length_data]))-np.array(np.log(SP500data[0:length_data-1])))
    GARCH('normal', SP500returns)
    GARCH('t', SP500returns)
    GARCH('GED', SP500returns)
    GARCH('EGB2', SP500returns)