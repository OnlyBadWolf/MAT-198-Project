import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import csv

import warnings
warnings.filterwarnings('ignore')



# Global variables
N = 0
y0 = ()





"""DERIVATIVE OF SIR-----------------------------------------------------------
This function calculates and define the derviatives for the data.
----------------------------------------------------------------------------"""
def deriv(y, t, L, k, t_0,  L_g, K, t_0g, alpha):
    S, E, I, R = y
    dSdt = -beta(t,L,k,t_0) * S * I / N
    dEdt = beta(t,L,k,t_0) * S * I / N - alpha * E
    dIdt = alpha * E - gamma(t,  L_g, K, t_0g) * I #beta(t,L,k,t_0) * S * I / N - gamma(t,  L_g, K, t_0g) * I
    dRdt = gamma(t,  L_g, K, t_0g) * I
    return dSdt, dEdt, dIdt, dRdt




"""CALCULATE TOTAL POPULATION IN THE US----------------------------------------
This function calculates the total population of the United States.
----------------------------------------------------------------------------"""
def totalPop():
    total_pop = 0
    with open("covid_county_population_usafacts(4-22).csv") as pop:
        reader_pop = csv.DictReader(pop)
        total_pop = sum (float(row["population"]) for row in reader_pop)
    return total_pop




"""CALCULATE TOTAL NUMBER OF CASES IN THE US-----------------------------------
This function calculates the total number of cases in the United States.
----------------------------------------------------------------------------"""
def totalNumberOfCases():
    # Get the total number of cinfirmed cases
    totConfirmed = []
    with open("covid_confirmed_usafacts(4-22).csv") as file:
        reader = csv.reader(file)
        
        # get the number of columns
        numDays = len(next(reader)) - 4
        
        totConfirmed = np.zeros(numDays)
        for row in reader:
            totConfirmed += [int(i) for i in row[4:]]
        
    return totConfirmed




"""CALCULATE TOTAL NUMBER OF DEATHS IN THE US----------------------------------
This function calculates the total number of deaths in the United States.
----------------------------------------------------------------------------"""
def totalNumberOfDeaths():
    # Get the total number of cinfirmed cases
    totConfirmed = []
    with open("covid_deaths_usafacts(4-22).csv") as file:
        reader = csv.reader(file)
        
        # get the number of columns
        numDays = len(next(reader)) - 4
        
        totDeaths = np.zeros(numDays)
        for row in reader:
            totConfirmed += [int(i) for i in row[4:]]
        
    return totDeaths




"""INTEGRATE THE SIR EQUATIONS OVER TIME---------------------------------------
This function integrates the SIR equation over time.
----------------------------------------------------------------------------"""
def integrateEquationsOverTime(deriv, t, L, k, t_0, L_g, K, t_0g, alpha):
    ret = odeint(deriv, y0, t, args=(L, k, t_0, L_g, K, t_0g, alpha))
    S, E, I, R = ret.T
    return S, E, I, R




"""PLOT THE SIR MODEL----------------------------------------------------------
This function plots the SIR Model.
----------------------------------------------------------------------------""" 
def plotsir(t, S, E, I, R):
    f, ax = plt.subplots(1,1,figsize=(10,4))
  
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'm', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, S+E+I+R, 'c--', alpha=0.7, linewidth=2, label='Total')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SEIR Model of COVID-19')


    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)   
    
    plt.show()
  



"""FITTING THE SIR MODEL-------------------------------------------------------
This function varies the parameters in the SIR Model.
----------------------------------------------------------------------------""" 
def fitter(t, L, k, t_0, L_g, K, t_0g, alpha):
    S, E, I, R = integrateEquationsOverTime(deriv, t, L, k, t_0, L_g, K, t_0g, alpha)
    return I




"""BETA FUNCTION---------------------------------------------------------------
This function calcuates the rate in which individuals are becoming infected, 
beta. We are using a logistics function because the rate of individuals 
being infected should decrease over time as there are less suseptible 
individuals.
----------------------------------------------------------------------------"""    
def beta(time, L, k, t_0):
    # t_0 is the value of the sigmoid's midpoint,
    # L is the curve's maximum value,
    # k is the logistic growth rate or steepness of the curve
    beta = L/(1+np.power(np.e, k*(time-t_0)))
    return beta




"""GASSIAN FUNCTION------------------------------------------------------
This calculates the gaussian function.
----------------------------------------------------------------------------"""
def gaussianG(t, mu, sig, phi):
    return phi * np.exp(-np.power(t - mu, 2) / (2 * np.power(sig, 2)))




"""GAMMA FUNCTION--------------------------------------------------------------
This function calculates the rate in which individuals recover from being 
infected, gamma. This uses a gaussian function because the rate of individuals 
should increase with the health care systems ablility to help people who are 
infected.
----------------------------------------------------------------------------"""
def gammaOld(time, mu, sig, phi):
    gamma = gaussianG(time, mu, sig, phi)
    return gamma


def gamma(time, L_g, K, t_0g):
    gamma = L_g/(1+np.power(np.e, -K*(time-t_0g)))
    return gamma



"""PLOTTING BETA FUNCTION------------------------------------------------------
This displays a graph of the beta function. It should look like a upside 
logistic function.
----------------------------------------------------------------------------"""
def plotBeta(times, L, k, t_0):
    fig, axsB = plt.subplots()
    
    axsB.set_title('Function of Beta')
    axsB.set_xlabel('Time (number of days)')
    axsB.set_ylabel('beta')
    axsB.plot(times, beta(times, L, k, t_0))
    plt.show()
    
    plt.clf()
    
    
    
    
"""PLOTTING GAMMA FUNCTION-----------------------------------------------------
This displays a graph of the gamma function. It should have part of a bell 
curve logistic function.
----------------------------------------------------------------------------"""
def plotGamma(times, L_g, K, t_0g):
    fig, axsG = plt.subplots()
    
    axsG.set_title('Function of Gamma')
    axsG.set_xlabel('Time (number of days)')
    axsG.set_ylabel('gamma')
    axsG.plot(times, gamma(times, L_g, K, t_0g))
    plt.show()
    plt.clf()
    
    
    

"""R_0 FUNCTION----------------------------------------------------------------
This function calculates R_0.
----------------------------------------------------------------------------"""
def calculateR_0(time, L, k, t_0, L_g, K, t_0g):
    R_0 = beta(time, L, k, t_0)/gamma(time, L_g, K, t_0g)
    return R_0





def plotR_0(time, L, k, t_0, L_g, K, t_0g):
    """PLOTTING R_0 FUNCTION-----------------------------------------------------
    This displays a graph of the gamma function. It should have part of a bell 
    curve logistic function.
    ----------------------------------------------------------------------------"""
    fig, axsR = plt.subplots()
    
    axsR.set_title('Function of R_0')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('R_0')
    axsR.plot(time, calculateR_0(time, L, k, t_0, L_g, K, t_0g))
    plt.show()
    plt.clf()



if __name__ == "__main__":
    total_con = totalNumberOfCases()
    total_deaths = totalNumberOfDeaths()
    
    # define constants
    N = totalPop()
    D = 14 # infections lasts 2 week on average (14 days)
    y0 = (N - 1, 1, 1, 0)  # initial conditions: one infected, rest susceptible
    moreTimes = np.linspace(0, 365-1, 365)
    alpha = 1/14
    
    # MAKING AN ARRAY
    times = np.linspace(0, len(total_con)-1, len(total_con)) # time (in days)
  
    mod = lmfit.Model(fitter)
    mod.set_param_hint('k', min = 0, max = 0.1)
    mod.set_param_hint('L', min=0, max = 10)
    mod.set_param_hint('L_g', min=0, max=0.16)
    mod.set_param_hint('K', min=0, max=0.3)
    mod.set_param_hint('t_0g', min=0)
    mod.set_param_hint('t_0', min=0)
    mod.set_param_hint('alpha', min = 0.06, max =0.2) # was man =0.07
    
    # mod.set_param_hint('mu', min= 200, max = 400)
    # mod.set_param_hint('sig', min=70, max =100)
    # mod.set_param_hint('phi', min = 2, max = 5)
    
    params = mod.make_params(verbose=True)
    result = mod.fit(total_con, 
                     params, 
                     t=times, 
                     L = 2, 
                     k=0.01, 
                     t_0 = 80, 
                     L_g = 0.14,
                     K = 0.1,
                     t_0g = 80,
                     alpha = 0.07)
                     #mu = 60, 
                     #sig = 80, 
                     #phi = 2) #beta=1.3, gamma=1.3)

    plotBeta(moreTimes, 
             result.best_values['L'], 
             result.best_values['k'], 
             result.best_values['t_0'])
    
    plotGamma(moreTimes, 
              result.best_values['L_g'], 
              result.best_values['K'], 
              result.best_values['t_0g'])
              # result.best_values['mu'], 
              # result.best_values['sig'], 
              # result.best_values['phi'])
    
    print(calculateR_0(moreTimes,
            result.best_values['L'], 
              result.best_values['k'], 
              result.best_values['t_0'],
              result.best_values['L_g'], 
              result.best_values['K'], 
              result.best_values['t_0g']))
    
    plotR_0(moreTimes,
            result.best_values['L'], 
              result.best_values['k'], 
              result.best_values['t_0'],
              result.best_values['L_g'], 
              result.best_values['K'], 
              result.best_values['t_0g'])
    
    #plt.plot(times, gaussianG(times, result.best_values['mu'], result.best_values['sig']))

    #print(gamma(moreTimes, result.best_values['mu'], result.best_values['sig'], result.best_values['phi']))

    result.plot()
    print(result.best_values)
    
    S, E, I, R = integrateEquationsOverTime(deriv, moreTimes,  
                                         result.best_values['L'], 
                                         result.best_values['k'],  
                                         result.best_values['t_0'],  
                                         result.best_values['L_g'], 
                                         result.best_values['K'], 
                                         result.best_values['t_0g'],
                                         alpha)
    
    print(max(totalNumberOfCases()))
    print(min(S[:92]),max(E[:92]),max(I[:92]),max(R[:92]))
    
    #PLOT SIR MODEL
    plotsir(moreTimes, S, E, I, R)
    