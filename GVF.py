import numpy

#from TileCoder import *
#from Coder import *
from StateRepresentation import *
import pickle

class GVF:
    def __init__(self, featureVectorLength, alpha, isOffPolicy, name = "GVF name"):
        #set up lambda, gamma, etc.
        self.name = name
        self.isOffPolicy = isOffPolicy
        self.numberOfFeatures = featureVectorLength
        self.lastState = 0
        self.lastObservation = 0
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.hWeights = numpy.zeros(featureVectorLength)
        self.hHatWeights = numpy.zeros(featureVectorLength)
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1

        self.alpha = (1.0 - 0.90) * alpha
        self.alphaH = 0.01 * self.alpha #Same alpha for H vector and each HHat vector

        self.alphaRUPEE = 5.0 * self.alpha
        self.betaNotUDE = self.alpha * 10.0
        #self.betaNotRUPEE = (1.0 - 0.90) * alpha * TileCoder.numberOfTilings / 30
        self.betaNotRUPEE = (1.0 - 0.90) * alpha * 9 / 30
        self.taoRUPEE = 0
        self.taoUDE = 0
        self.movingtdEligErrorAverage = 0 #average of TD*elig*hHat
        self.lastAction = 0

        self.tdVariance = 0
        self.averageTD = 0
        self.i = 1

        self.meta_weights = np.ones(self.numberOfFeatures) * np.log(self.alpha)
        self.meta_weight_trace = np.ones(self.numberOfFeatures)
        self.meta_step_size = 0.1

    def update_meta_traces(self, phi, td_error):
        """Updates the meta-traces for TDBID; is an accumulating trace of recent weight updates.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step.
        """
        raise NotImplementedError()

    def update_meta_weights(self, phi, td_error):
        """Updates the meta-weights for TIDBD; these are used to set the step-sizes.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step.
        """
        raise NotImplementedError()

    def update_normalizer_accumulation(self, phi, td_error):
        """Tracks the size of the meta-weight updates.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step."""
        raise NotImplementedError()

    def get_effective_step_size(self, gamma, phi, phi_next):
        """Returns the effective step-size for a given time-step
        Args:
            phi (ndarray): the last feature vector; represents s_t
            phi_next (ndarray): feature vector for state s_{t+1}
            gamma (float): discount factor
        Returns:
            effective_step_size (float): the amount by which the error was reduced on a given example.
        """
        delta_phi = (gamma * phi_next - phi)
        return np.dot(-(np.exp(self.beta) * self.z), delta_phi)

    def normalize_step_size(self, gamma, phi, phi_next):
        """Calculates the effective step-size and normalizes the current step-size by that amount.
        Args:
            gamma (float): discount factor
            phi (ndarray): feature vector for state s_t
            phi_next (ndarray): feature vector for state s_{t+1}"""
        effective_step_size = self.get_effective_step_size(gamma, phi, phi_next)
        m = np.maximum(effective_step_size, 1.)
        self.beta /= np.log(m)

    def tidbid(self, phi, phi_next, gamma, td_error):
        """Using the feature vector for s_t and the current TD error, performs TIDBD and updates step-sizes.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            phi_next (ndarray): feature vector for state s_{t+1}
            gamma (float): discount factor
            td_error (float): the temporal-difference error for the current time-step.
            
        """
        self.update_normalizer_accumulation(phi,td_error)
        self.update_meta_weights(phi, td_error)
        self.normalize_step_size(gamma, phi, phi_next)
        self.update_meta_traces(phi, td_error)
        self.alpha = np.exp(self.meta_weights)

    def calculate_step_size(self):
        """Calculates the current alpha value using the meta-weights
        Returns:
             None
        """
        self.alpha = np.exp(self.beta)

    def saveWeightsToPickle(self, file):
        pickleDict = {'weights': self.weights, 'hWeights': self.hWeights, 'hHatWeights': self.hHatWeights}
        with open(file, 'wb') as outfile:
            pickle.dump(pickleDict, outfile)

    def readWeightsFromPickle(self, file):
        with open(file, 'rb') as inFile:
            error = False
            pickleDict = pickle.load(inFile)
            print("read pickle dictionary")

            self.weights = pickleDict['weights']
            self.hWeights = pickleDict['hWeights']
            self.hHatWeights = pickleDict['hHatWeights']

            print("Read weights. ")

    def reset(self):
        self.lastState = 0
        self.lastObservation = 0
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.hWeights = numpy.zeros(self.numberOfFeatures)
        self.hHatWeights = numpy.zeros(self.numberOfFeatures)
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1

        self.movingtdEligErrorAverage = 0 #average of TD*elig*hHat
        self.lastAction = 0

        self.tdVariance = 0
        self.averageTD = 0
        self.i = 1
    """
    gamma, cumulant, and policy functions can/should be overiden by the specific instantiation of the GVF based on the intended usage.
    """
    def gamma(self, state):
        return 0.0

    def cumulant(self, state):
        return 1

    def policy(self, state):
        #To be overwritten based on GVF's intended behavior if off policy. Otherwise 1 means on policy
        return 'F'

    def lam(self, state):
        return 0.90

    def rho(self, action, state):
        targetAction = self.policy(state)
        if targetAction == action:
            return 1
        else:
            return 0

    def learn(self, lastState, action, newState):
        #print("Learning")
        """
        print("My weights (length = " + str(len(self.weights)) + ")")
        print(self.weights)
        print("Last state (length = " + str(len(lastState)) + ")")
        print(lastState)
        print("action:")
        print(action)
        print("New state (length = " + str(len(newState)) + ")")
        print(newState)
        """
        if self.isOffPolicy:
            self.gtdLearn(lastState, action, newState)
        else:
            self.tdLearn(lastState, action, newState)

    def gtdLearn(self, lastState, action, newState):

        #print("GVF name: " + str(self.name))
        #print("For (" + str(lastState.colorLookingAt) +  ", " + str(lastState.didBump) +  ") to (" + str(newState.colorLookingAt) + ", " + str(newState.didBump) + ")")
        pred = self.prediction(lastState)
        #print("--- Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed) + " before learning: " + str(pred))
        #print("--- Prediction before: " + str(pred))

        #print("action")
        #print(action)
        zNext = self.cumulant(newState)
        #print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState)
        #print("gammaNext: " + str(gammaNext))
        lam = self.lam(newState)
        #print("gammaLast: " + str(self.gammaLast))
        '''
        #TODO REMOVE AFTER TESTING
        changedOldIndexes = []
        changedNewIndexes = []
        if self.name == 'TA':
            for i in range( PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES -1):
                if lastState[i] >0:
                    lastState[i]=0
                    changedOldIndexes.append(i)
                if newState[i]>0:
                    newState[i] = 0
                    changedNewIndexes.append(i)
                lastState[i] = 0
                newState[i]=0
        '''
        #print("lambda: " + str(lam))
        rho = self.rho(action, lastState)
        #print("rho: " + str(rho))
        self.eligibilityTrace = rho * (self.gammaLast * lam * self.eligibilityTrace + lastState)
        newStateValue = 0.0
        if not newState  is None:
            newStateValue = numpy.inner(newState, self.weights)
        tdError = zNext + gammaNext * newStateValue - numpy.inner(lastState, self.weights)


        """
        if (zNext >0):
            print("Selph.alphaH: " + str(self.alphaH))
            print("tdError: " + str(tdError))
            print("Eligibility length: " + str(len(self.eligibilityTrace)))
            print("hweight length: " + str(len(self.hWeights)))
            print("last state length: " + str(len(lastState)))
            print("lam: " + str(lam))
            print("rho: " + str(rho))
        """
        updateH = self.alphaH * (tdError * self.eligibilityTrace - (numpy.inner(self.hWeights, lastState)) * lastState)

        self.hWeights = self.hWeights + updateH

        """
        #update Rupee
        self.hHatWeights = self.hHatWeights + self.alphaRUPEE * (tdError * self.eligibilityTrace - (numpy.inner(self.hHatWeights, lastState)) * lastState)
        #print("tao before: " + str(self.tao))
        self.taoRUPEE = (1.0 - self.betaNotRUPEE) * self.taoRUPEE + self.betaNotRUPEE
        #print("tao after: " + str(self.tao))

        betaRUPEE = self.betaNotRUPEE / self.taoRUPEE
        #print("beta: " + str(beta))
        self.movingtdEligErrorAverage = (1.0 - betaRUPEE) * self.movingtdEligErrorAverage + betaRUPEE * tdError * self.eligibilityTrace

        #update UDE
        self.taoUDE = (1.0 - self.betaNotUDE) * self.taoUDE + self.betaNotUDE
        betaUDE = self.betaNotUDE / self.taoUDE

        oldAverageTD = self.averageTD
        #print("Old averageTD:" + str(oldAverageTD))


        self.averageTD = (1.0 - betaUDE) * self.averageTD + betaUDE * tdError
        #print("New AverageTD: " + str(self.averageTD))
        #print("tdvariance before: " + str(self.tdVariance))
        self.tdVariance = ((self.i - 1) * self.tdVariance + (tdError - oldAverageTD) * (tdError - self.averageTD)) / self.i
        #print("td variance after: " + str(self.tdVariance))
        """
        self.i = self.i + 1

        upWeights = self.alpha * (tdError * self.eligibilityTrace - gammaNext * (1-lam)  * (numpy.inner(self.eligibilityTrace, self.hWeights) * newState))
        if (zNext >0):
            t = 0
            #print("upWeights: ")
            #print(upWeights)
            for w in upWeights:
                if w>0:
                    t = t + 1
                    #w i 0.0025
            #print("Total updates with weight: " + str(t))

            """
            print("--")
            t =0
            for w in self.weights:
                if w> 0:
                    t = t+1
            print("Total weights with value before: " + str(t))
            """
        self.weights = self.weights + upWeights

        pred = self.prediction(lastState)
        #print("Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed)  + " after learning: " + str(pred))

        #rupee = self.rupee()
        #print("Rupee: " + str(rupee))

        #ude = self.ude()
        #print("UDE: " + str(ude))

        self.gammaLast = gammaNext

        '''
        #TODO REMOVE AFTER TESTING
        if self.name == 'TA':
            for idx in changedOldIndexes:
                lastState[idx] = 1
            for idx in changedNewIndexes:
                newState[idx] = 1
        '''

    def tdLearn(self, lastState, action, newState):
        print("!!!!! LEARN  !!!!!!!")
        print("GVF name: " + str(self.name))
        #print("For (" + str(lastState.encoder) +  ", " + str(lastState.speed) +  ") to (" + str(newState.encoder) + ", " + str(newState.speed) + ")")
        pred = self.prediction(lastState)
        #print("--- Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed) + " before learning: " + str(pred))

        #print("alpha: " + str(self.alpha))

        #print("action")
        #print(action)

        zNext = self.cumulant(newState)
        #print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState)
        #print("gammaNext: " + str(gammaNext))
        lam = self.lam(newState)
        #print("gammaLast: " + str(self.gammaLast))

        #print("lambda: " + str(lam))
        self.eligibilityTrace = self.gammaLast * lam * self.eligibilityTrace + lastState

        tdError = zNext + gammaNext * numpy.inner(newState, self.weights) - numpy.inner(lastState, self.weights)

        #print("tdError: " + str(tdError))

        #update Rupee
        self.hHatWeights = self.hHatWeights + self.alphaRUPEE * (tdError * self.eligibilityTrace - (numpy.inner(self.hHatWeights, lastState)) * lastState)
        #print("tao before: " + str(self.tao))
        self.taoRUPEE = (1.0 - self.betaNotRUPEE) * self.taoRUPEE + self.betaNotRUPEE
        #print("tao after: " + str(self.taoRUPEE))

        betaRUPEE = self.betaNotRUPEE / self.taoRUPEE
        #print("beta: " + str(beta))
        self.movingtdEligErrorAverage =(1.0 - betaRUPEE) * self.movingtdEligErrorAverage + betaRUPEE * tdError * self.eligibilityTrace


        #update UDE
        self.taoUDE = (1.0 - self.betaNotUDE) * self.taoUDE + self.betaNotUDE
        betaUDE = self.betaNotUDE / self.taoUDE
        oldAverageTD = self.averageTD
        self.averageTD = (1.0 - betaUDE) * self.averageTD + betaUDE * tdError
        self.tdVariance = ((self.i - 1) * self.tdVariance + (tdError - oldAverageTD) * (tdError - self.averageTD)) / self.i
        self.i = self.i + 1

        self.weights = self.weights + self.alpha * tdError * self.eligibilityTrace

        pred = self.prediction(lastState)
        #print("Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed)  + " after learning: " + str(pred))
        rupee = self.rupee()

        #print("Rupee: " + str(rupee))

        ude = self.ude()
        #print("UDE: " + str(ude))

        self.gammaLast = gammaNext

    def prediction(self, stateRepresentation):
        return numpy.inner(self.weights, stateRepresentation)

    def rupee(self):
        return numpy.sqrt(numpy.absolute(numpy.inner(self.hHatWeights, self.movingtdEligErrorAverage)))

    def ude(self):
        return numpy.absolute(self.averageTD / (numpy.sqrt(self.tdVariance) + 0.000001))