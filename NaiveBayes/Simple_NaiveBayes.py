""" Simple Training Set
    Height   Weight   Foot_Size  Gender	
    -----------------------------------
      6.00     180      12        Male
      5.92     190      11        Male
      5.58     170      12        Male 
      5.92     165      10        Male
      5.00     100      6         Female
      5.50     150      8         Female
      5.42     130      7         Female
      5.75     150      9         Female

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
    print(class_probabilities)
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])

        print(row_indices)
        print(subset)
        print(likelihoods)

    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
    print(likelihoods)
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             print(new_sample)
             print(relative_values)
             print(relative_values.keys())
             print(new_sample[i])

             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
                 
             else:
                 class_probability *= 0
             print(class_probability)
             results[cls] = class_probability
             
    if results[0] > results[1]:
        print ('Male')
    else:
        print('Female')
    print(results)
 
if __name__ == "__main__":
    """For the outcome we set as 0 for male and 1 female. """
    training   = np.asarray(((6.00, 180, 12),
                             (5.92, 190, 11),
                             (5.58, 170, 12),
                             (5.92, 165, 10),
                             (5.00, 100, 6),
                             (5.50, 150, 8),
                             (5.42, 130, 7),
                             (5.75, 150, 9)));
    outcome    = np.asarray((0,0,0,0,1,1,1,1))
    
    new_sample = np.asarray((6.00,130,8))
    naive_bayes(training, outcome, new_sample)
