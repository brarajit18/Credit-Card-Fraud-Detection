# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

titles = ['TP','TN','FP','FN']
fig_titles = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
fig = plt.figure(figsize=(12, 5))
#Plot 1
iiddx2 = 0
y_pos = [1, 2, 3]
objects = ['Naive Bayes','KNN','SVM']
performance = np.zeros([3,1])
iddxx=0
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=1
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=2
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]

sub1 = fig.add_subplot(221)
sub1.bar(y_pos, performance, align='center', alpha=0.5)
sub1.set_xticks(y_pos)
sub1.set_xticklabels(objects)
sub1.set_ylabel(fig_titles[iiddx2])
iiddx2 = iiddx2 + 1

#Plot 2
performance = np.zeros([3,1])
iddxx=0
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=1
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=2
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]

sub2 = fig.add_subplot(222)
sub2.bar(y_pos, performance, align='center', alpha=0.5)
sub2.set_xticks(y_pos)
sub2.set_xticklabels(objects)
sub2.set_ylabel(fig_titles[iiddx2])
iiddx2 = iiddx2 + 1

#Plot 3
performance = np.zeros([3,1])
iddxx=0
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=1
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=2
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]

sub3 = fig.add_subplot(223)
sub3.bar(y_pos, performance, align='center', alpha=0.5)
sub3.set_xticks(y_pos)
sub3.set_xticklabels(objects)
sub3.set_ylabel(fig_titles[iiddx2])
iiddx2 = iiddx2 + 1


#Plot 4
performance = np.zeros([3,1])
iddxx=0
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=1
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]
iddxx=2
performance[iddxx]=Overall_results[titles[iiddx2]][iiddx]

sub4 = fig.add_subplot(224)
sub4.bar(y_pos, performance, align='center', alpha=0.5)
sub4.set_xticks(y_pos)
sub4.set_xticklabels(objects)
sub4.set_ylabel(fig_titles[iiddx2])
iiddx2 = iiddx2 + 1

plt.show()
