""" Simple Training Set
    Chills   RunnyNose   Headache  Fever  Flu
    -----------------------------------------
      Y         N          Mild     Y      N
      Y         Y           No      N      Y
      Y         N         Strong    Y      Y
      N         Y          Mild     Y      Y
      N         N           No      N      N
      N         Y         Strong    Y      Y
      N         Y         Strong    N      N
      Y         Y          Mild     Y      Y

    ------------------------------------------

    """



import numpy as np
from collections import Counter, defaultdict

def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob

def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)    
    rows, cols  = np.shape(training)
    
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    
    class_probabilities = occurrences(outcome) 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])

    

    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]

             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
                 
             else:
                 class_probability *= 0
             results[cls] = class_probability
             
    if results[0] > results[1]:
        print ('No Flu')
    else:
        print('Flu')
    print(results)
 
if __name__ == "__main__":
    """Foremost, we build the training data.
    For the features, we provide label 1 to all Y
    (ie, the symptom was present in the patient)
    and 0 to all N (ie, the symptom was not present
    in the patient). Also, in the case of the feature
    “headache”, no headache = 0, mild = 1 and strong = 2.
    The outcome also follows the same pattern and
    thus, no flu = 0 and flu = 1. """
    training   = np.asarray(((1,0,1,1),
                             (1,1,0,0),
                             (1,0,2,1),
                             (0,1,1,1),
                             (0,0,0,0),
                             (0,1,2,1),
                             (0,1,2,0),
                             (1,1,1,1)));
    outcome    = np.asarray((0,1,1,1,0,1,0,1))
    
    new_sample = np.asarray((1,0,1,1))
    naive_bayes(training, outcome, new_sample)
