import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import csv

import warnings
warnings.filterwarnings('ignore')


"""To-do List:
    - figure out how to get a better fit
    - model upto March 15th before shelter in place
    """


# Global variables
N = 0
y0 = ()





"""DERIVATIVE OF SIR-----------------------------------------------------------
This function calculates and define the derviatives for the data.
----------------------------------------------------------------------------"""
def deriv(y, t, L, k, t_0, gamma, alpha, delta): #L_g, K, t_0g
    S, E, I, R, D = y
    dSdt = -beta(t,L,k,t_0) * S * I / N
    dEdt = beta(t,L,k,t_0) * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I #beta(t,L,k,t_0) * S * I / N - gamma(t,  L_g, K, t_0g) * I
    dRdt = gamma *(1 - delta)* I
    dDdt = gamma * delta * I
    return dSdt, dEdt, dIdt, dRdt, dDdt




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
    # Get the total number of confirmed cases
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
    # Get the total number of dead individuals from COVID-19
    totDeaths = []
    with open("covid_deaths_usafacts(4-22).csv") as file:
        reader = csv.reader(file)
        
        # get the number of columns
        numDays = len(next(reader)) - 4
        
        totDeaths = np.zeros(numDays)
        for row in reader:
            totDeaths += [int(i) for i in row[4:]]
        
    return totDeaths




"""INTEGRATE THE SIR EQUATIONS OVER TIME---------------------------------------
This function integrates the SIR equation over time.
----------------------------------------------------------------------------"""
def integrateEquationsOverTime(deriv, t, L, k, t_0, gamma, alpha,  delta):
    ret = odeint(deriv, y0, t, args=(L, k, t_0, gamma, alpha,  delta))
    S, E, I, R, D = ret.T
    return S, E, I, R, D




"""PLOT THE SEIRD MODEL--------------------------------------------------------
This function plots the SEIRD Model.
----------------------------------------------------------------------------""" 
def plotsir(t, S, E, I, R, D):
    f, ax = plt.subplots(1,1,figsize=(10,4))
  
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'm', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'r', alpha=0.7, linewidth=2, label='Died')
    ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_ylim(0, 1200000)
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
    
    plt.clf()
  



"""FITTING THE SEIRD MODEL-----------------------------------------------------
This function varies the parameters in the SEIRD Model.
----------------------------------------------------------------------------""" 
def fitter(t, L, k, t_0, gamma, alpha, delta):
    S, E, I, R, D = integrateEquationsOverTime(deriv, t, L, k, t_0, gamma, alpha, delta)
    return np.concatenate([I, D])




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



"""GAMMA FUNCTION------------------------------------------------------
This function calcuates the rate in which individuals recover, 
gamma. We are using a logistics function because the rate of individuals 
recovering should increase over time as there are more infected 
individuals.
----------------------------------------------------------------------------"""
def gamma(time, L_g, K, t_0g):
    gamma = L_g/(1+np.power(np.e, -K*(time-t_0g)))
    return gamma



"""DELTA FUNCTION--------------------------------------------------------------
This function calcuates the rate in which individuals dying, 
delta. We are using a logistics function because the rate of individuals 
dying should decrease over time as we get better at treating.
----------------------------------------------------------------------------"""
def delta(time, Ld, kd, t_0d):
    gamma = Ld/(1+np.power(np.e, -kd*(time-t_0d)))
    return gamma


"""PLOTTING DELTA FUNCTION-----------------------------------------------------
This displays a graph of the delta function. It should have part of a bell 
curve logistic function.
----------------------------------------------------------------------------"""
def plotDelta(times, Ld, kd, t_0d):
    fig, axsG = plt.subplots()
    
    axsG.set_title('Function of Delta')
    axsG.set_xlabel('Time (number of days)')
    axsG.set_ylabel('delta')
    axsG.plot(times, delta(times, Ld, kd, t_0d))
    plt.show()
    plt.clf()




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
This displays a graph of the gamma function. It should look like as logistic 
function.
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
def calculateR_0(time, L, k, t_0, gamma):
    R_0 = beta(time, L, k, t_0)/gamma
    return R_0




"""PLOTTING R_0 FUNCTION-----------------------------------------------------
This displays a graph of the gamma function. It should have part of a bell 
curve logistic function.
----------------------------------------------------------------------------"""
def plotR_0(time, L, k, t_0, gamma):
    fig, axsR = plt.subplots()
    
    axsR.set_title('Function of R_0')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('R_0')
    
    axsR.plot(time, calculateR_0(time, L, k, t_0, gamma))
    
    plt.show()
    plt.clf()
    
    
    
    
"""PLOT THE BEST FIT OF INFECTED-----------------------------------------------
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------""" 
def plotBestFitInfected(t, I, total_con, residualBitch):
    fig, axR = plt.subplots()
    
    axR.plot(t, residualBitch)
    axR.axhline(0, linestyle='--')

    axR.set_title('Residual of Infected')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (days)')
    
    plt.show()
    plt.clf()
    
    f, axI = plt.subplots()

    axI.scatter(t, total_con, s=4, label='data')
    axI.plot(t, I, 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axI.set_xlabel('Time (days)')
    axI.set_ylabel('Population')
    axI.set_title('Infected Population')

    axI.yaxis.set_tick_params(length=0)
    axI.xaxis.set_tick_params(length=0)
    axI.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axI.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axI.spines[spine].set_visible(False)   
    
    plt.show()
    plt.clf()
    
    
"""PLOT THE BEST FIT OF DEAD---------------------------------------------------
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------""" 
def plotBestFitDied(t, D, total_deaths, residualBitch):
    fig, axR = plt.subplots()
    #axR = fig.add_subplot(411, anchor=(0, 1))
    axR.plot(t, residualBitch)
    axR.axhline(0, linestyle='--')

    axR.set_title('Residual of Died')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (days)')
    
    plt.show()
    plt.clf()
    
    
    fig, axD = plt.subplots()
    
    axD.scatter(t, total_deaths, s=4, label='data')
    axD.plot(t, D, 'y', label='best fit')
    
    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axD.set_xlabel('Time (days)')
    axD.set_ylabel('Population')
    axD.set_title('Population Died from COVID-19')

    axD.yaxis.set_tick_params(length=0)
    axD.xaxis.set_tick_params(length=0)
    axD.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axD.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axD.spines[spine].set_visible(False)   
    
    plt.show()
    plt.clf() 



if __name__ == "__main__":
    total_con = totalNumberOfCases()
    total_deaths = totalNumberOfDeaths()
    
    # define constants
    N = totalPop()
    #D = 14 # infections lasts 2 week on average (14 days)
    y0 = (N - 2, 1, 1, 0, 0)  # initial conditions: one infected, rest susceptible
    moreTimes = np.linspace(0, 365-1, 365)
    #alpha = 1/6 # incubation rate
    #gamma = 1/14 # rate of recovery

    # MAKING AN ARRAY
    times = np.linspace(0, len(total_con)-1, len(total_con)) # time (in days)
  
    mod = lmfit.Model(fitter)
    mod.set_param_hint('k', min = 0, max = 0.15)
    mod.set_param_hint('L', min=0, max = 6)
    mod.set_param_hint('t_0', min=0, max=365)
    mod.set_param_hint('gamma', min=0.066, max=0.2) # sick for 2 weeks up to 6
    mod.set_param_hint('alpha', min = 0.05, max=0.2) # 5-6 days up to 14 days
    mod.set_param_hint('delta', min=0, max = 1) 
    

    # Puts Total number of case and deaths in 1 array to calculate best fit
    data = []
    data = np.concatenate([total_con, total_deaths])
     
    params = mod.make_params(verbose=True)
    result = mod.fit(data,
                     params, 
                     t=times, 
                     L = 2, 
                     k = 0.0005, 
                     t_0 = 52.100165375262584, 
                     gamma = 0.074,
                     alpha = 0.16,
                     delta = 0.006)

    plotBeta(moreTimes, 
             result.best_values['L'], 
             result.best_values['k'], 
             result.best_values['t_0'])

    
    plotR_0(moreTimes,
              result.best_values['L'], 
              result.best_values['k'], 
              result.best_values['t_0'],
              result.best_values['gamma'])
    
    #result.plot()

    # Prints L, k, t_0, gamma, alpha, delta
    print(result.best_values)
    
    
    #Integrate SEIRD
    S, E, I, R, D = integrateEquationsOverTime(deriv, moreTimes,  
                                         result.best_values['L'], 
                                         result.best_values['k'],  
                                         result.best_values['t_0'],  
                                         result.best_values['gamma'],
                                         result.best_values['alpha'],
                                         result.best_values['delta']) 
                                         #result.best_values['delta'])
    
    # Create an numpy array
    residualBitch = result.residual
    
    # Plot Residual and Best Fit
    plotBestFitInfected(times, I[:122], total_con, residualBitch[:122])
    plotBestFitDied(times, D[:122], total_deaths, residualBitch[122:])

    
    print('Population of the US:', N)
    print('Total Number of Deaths:', max(total_deaths))
    print('Total Number of Cases:', max(totalNumberOfCases()))
    print('Suspetible:', min(S[:len(times)]))
    print('Exposed:', max(E[:len(times)]))
    print('Infected:', max(I[:len(times)]))
    print('Recovered:', max(R[:len(times)])) # should be somewhere around 300,000
    print('Dead:',max(D[:len(times)])) # should equal total dead
    
    total = S+I+E+R+D
    
    print('Total:', min(total)) # should equal total population
    
    #PLOT SIR MODEL
    plotsir(moreTimes, S, E, I, R, D)
    
    #PLOT WITHOUT QUARENTINE (doesn't work)
    #plotsir(times, S[:53], E[:53], I[:53], R[:53], D[:53])
    