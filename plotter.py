#!/usr/bin/python
# coding: utf-8
from __future__ import print_function
import pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import numpy as np
from os import system


with open('results/experiment_results.pkl', "rb") as f:
    results = pickle.load(f)


exp_rupee = results['predictive_rupee']
exp_ude = results['predictive_ude']
exp_prediction = results['predictive_predictions']
track_rupee = results['tracking_rupee']
track_ude = results['tracking_ude']
track_prediction = results['tracking_predictions']
true_values = results['environment_values']

for i in range(len(exp_rupee)):
    exp_prediction[i] = np.average(exp_prediction[i], axis=0)
    exp_ude[i] = np.average(exp_ude[i], axis=0)
    exp_rupee[i] = np.average(exp_rupee[i], axis=0)
    track_prediction[i] = np.average(track_prediction[i], axis=0)
    track_ude[i] = np.average(track_ude[i], axis=0)
    track_rupee[i] = np.average(track_rupee[i], axis=0)

print("plotting")

ax = pyplot.gca()
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)

# pyplot.plot(exp_rupee[0], alpha=0.4, label="Prediction Touch")
# pyplot.plot(track_rupee[0], label="Tracking Touch")

pyplot.plot([np.sum(exp_rupee[0][:j]) for j in range(len(exp_rupee[i]))], alpha=0.4, label="predictor")
pyplot.plot([np.sum(track_rupee[0][:j]) for j in range(len(exp_rupee[i]))], alpha=0.4, label="tracker")


pyplot.ylabel("RUPEE")
pyplot.xlabel("Time-steps")
pyplot.legend()
pyplot.show()
print(np.sum(exp_rupee[i], axis=0), np.sum(track_rupee[i], axis=0))
# pyplot.plot()



# pyplot.plot([sum(exp_rupee[i][:j]) for j in range(len(exp_rupee[i]))], alpha=0.4, label="predictor")
# pyplot.plot([sum(track_rupee[i][:j]) for j in range(len(exp_rupee[i]))], alpha=0.4, label="tracker")

pyplot.clf()

ax = pyplot.gca()
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)

# pyplot.plot(track_rupee[1], label="Tracking Touch")
# pyplot.plot(exp_rupee[1], label="Prediction Touch")
pyplot.plot([np.sum(exp_rupee[1][:j], axis=0) for j in range(len(exp_rupee[i]))], alpha=0.4, label="predictor")
pyplot.plot([np.sum(track_rupee[1][:j], axis=0) for j in range(len(exp_rupee[i]))], alpha=0.4, label="tracker")

pyplot.ylabel("RUPEE")
pyplot.xlabel("Time-steps")
pyplot.legend()
pyplot.show()

pyplot.clf()

ax = pyplot.gca()
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)

pyplot.plot(track_prediction[0], label="Tracking Touch")
pyplot.plot(exp_prediction[0], label="Prediction Touch")
pyplot.plot(true_values[0][:25000], color='black', linestyle=':')
pyplot.ylabel("RUPEE")
pyplot.xlabel("Time-steps")
pyplot.legend()
pyplot.show()

# system('say your experiment is finished')
