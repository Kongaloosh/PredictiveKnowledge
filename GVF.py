#from TileCoder import *
#from Coder import *
from pysrc.function_approximation.StateRepresentation import *
import pickle


class GVF:
    def __init__(self, featureVectorLength, alpha, is_off_policy, name ="GVF name"):
        #set up lambda, gamma, etc.
        self.name = name
        self.isOffPolicy = is_off_policy
        self.numberOfFeatures = featureVectorLength
        self.lastState = 0
        self.lastObservation = 0
        self.weights = np.zeros(self.numberOfFeatures)
        self.hWeights = np.zeros(featureVectorLength)
        self.hHatWeights = np.zeros(featureVectorLength)
        self.eligibilityTrace = np.zeros(self.numberOfFeatures)
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
        self.meta_normalizer_trace = np.zeros(self.numberOfFeatures)
        self.meta_step_size = 0.1

    def update_meta_traces(self, td_error):
        """Updates the meta-traces for TDBID; is an accumulating trace of recent weight updates.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step.
        """
        self.h += self.alpha * td_error * self.eligibilityTrace

    def update_meta_weights(self, phi, td_error):
        """Updates the meta-weights for TIDBD; these are used to set the step-sizes.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step.
        """
        self.meta_weights += phi * self.meta_step_size * td_error * self.meta_weight_trace

    def update_normalizer_accumulation(self, phi, td_error):
        """Tracks the size of the meta-weight updates.
        Args:
            phi (ndarray): the last feature vector; represents s_t
            td_error (float): the temporal-difference error for the current time-step."""
        delta_phi = -phi
        update = np.abs(td_error * delta_phi * self.meta_weight_trace)
        tracker = np.exp(self.meta_weights) * self.eligibilityTrace * delta_phi
        self.meta_normalizer_trace = np.maximum(
            np.abs(update),
            self.meta_normalizer_trace + (1./self.tau) * tracker * (np.abs(update) - self.meta_normalizer_trace)
        )

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
        return np.dot(-(np.exp(self.meta_weights) * self.eligibilityTrace), delta_phi)

    def normalize_step_size(self, gamma, phi, phi_next):
        """Calculates the effective step-size and normalizes the current step-size by that amount.
        Args:
            gamma (float): discount factor
            phi (ndarray): feature vector for state s_t
            phi_next (ndarray): feature vector for state s_{t+1}"""
        effective_step_size = self.get_effective_step_size(gamma, phi, phi_next)
        m = np.maximum(effective_step_size, 1.)
        self.meta_weights /= np.log(m)

    def tdbid(self, phi, phi_next, gamma, td_error):
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
        self.calculate_step_size()
        self.update_meta_traces(phi, td_error)

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
        self.weights = np.zeros(self.numberOfFeatures)
        self.hWeights = np.zeros(self.numberOfFeatures)
        self.hHatWeights = np.zeros(self.numberOfFeatures)
        self.eligibilityTrace = np.zeros(self.numberOfFeatures)
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
        raise NotImplementedError("GVF {0} has no gamma.".format(self.name))

    def cumulant(self, state):
        raise NotImplementedError("GVF {0} has no cumulant.".format(self.name))

    def policy(self, state):
        raise NotImplementedError("GVF {0} has no policy.".format(self.name))

    def lam(self, state):
        return 0.90

    def rho(self, action, state):
        targetAction = self.policy(state)
        if targetAction == action:
            return 1
        else:
            return 0

    def learn(self, lastState, action, newState):
        if self.isOffPolicy:
            self.gtdLearn(lastState, action, newState)
        else:
            self.tdLearn(lastState, action, newState)

    def gtdLearn(self, lastState, action, newState):
        """GTD Learning

        """
        pred = self.prediction(lastState)
        zNext = self.cumulant(newState)
        gammaNext = self.gamma(newState)
        lam = self.lam(newState)
        rho = self.rho(action, lastState)
        self.eligibilityTrace = rho * (self.gammaLast * lam * self.eligibilityTrace + lastState)
        newStateValue = 0.0
        if not newState  is None:
            newStateValue = np.inner(newState, self.weights)
        tdError = zNext + gammaNext * newStateValue - np.inner(lastState, self.weights)
        updateH = self.alphaH * (tdError * self.eligibilityTrace - (np.inner(self.hWeights, lastState)) * lastState)
        self.hWeights = self.hWeights + updateH
        self.i = self.i + 1

        upWeights = self.alpha * (tdError * self.eligibilityTrace - gammaNext * (1-lam)  * (np.inner(self.eligibilityTrace, self.hWeights) * newState))
        if (zNext >0):
            t = 0
            for w in upWeights:
                if w>0:
                    t = t + 1
        self.weights = self.weights + upWeights
        pred = self.prediction(lastState)
        self.gammaLast = gammaNext



    def tdLearn(self, lastState, action, newState):
        pred = self.prediction(lastState)
        zNext = self.cumulant(newState)
        gammaNext = self.gamma(newState)
        lam = self.lam(newState)
        self.eligibilityTrace = self.gammaLast * lam * self.eligibilityTrace + lastState
        tdError = zNext + gammaNext * np.inner(newState, self.weights) - np.inner(lastState, self.weights)
        self.hHatWeights = self.hHatWeights + self.alphaRUPEE * (tdError * self.eligibilityTrace - (np.inner(self.hHatWeights, lastState)) * lastState)
        self.taoRUPEE = (1.0 - self.betaNotRUPEE) * self.taoRUPEE + self.betaNotRUPEE
        betaRUPEE = self.betaNotRUPEE / self.taoRUPEE
        self.movingtdEligErrorAverage =(1.0 - betaRUPEE) * self.movingtdEligErrorAverage + betaRUPEE * tdError * self.eligibilityTrace
        self.taoUDE = (1.0 - self.betaNotUDE) * self.taoUDE + self.betaNotUDE
        betaUDE = self.betaNotUDE / self.taoUDE
        oldAverageTD = self.averageTD
        self.averageTD = (1.0 - betaUDE) * self.averageTD + betaUDE * tdError
        self.tdVariance = ((self.i - 1) * self.tdVariance + (tdError - oldAverageTD) * (tdError - self.averageTD)) / self.i
        self.i = self.i + 1
        self.weights = self.weights + self.alpha * tdError * self.eligibilityTrace
        pred = self.prediction(lastState)
        rupee = self.rupee()
        ude = self.ude()
        self.gammaLast = gammaNext

    def prediction(self, stateRepresentation):
        return np.inner(self.weights, stateRepresentation)

    def rupee(self):
        return np.sqrt(np.absolute(np.inner(self.hHatWeights, self.movingtdEligErrorAverage)))

    def ude(self):
        return np.absolute(self.averageTD / (np.sqrt(self.tdVariance) + 0.000001))