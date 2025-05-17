import numpy as np

def H1(e):
    # Distribution: [1 - 1.5e, e/2, e/2, e/2]
    p0 = 1 - 1.5*e
    p1 = e/2
    probs = [p0] + [p1]*3
    return -sum([p*np.log2(p) for p in probs if p>0]) #some guessing make p=0 which is und.

def h(e):
    # Distribution: [1-e,e]
    p0 = 1-e
    p1 = e
    probs = [p0,p1]
    return -sum([p*np.log2(p) for p in probs if p>0])

# Solve for H(...)=1
# ees = np.linspace(0, 2/3, 10000)
# vals = [abs(H(x)-1) for x in ees]
# best_e = ees[np.argmin(vals)]
# print(best_e)

#Solve for h(...)=1/2
ees = np.linspace(0,1,10000)
vals = [abs(1-2*h(x)) for x in ees]
best_e = ees[np.argmin(vals)]
print(best_e)


