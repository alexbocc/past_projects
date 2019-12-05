#!/usr/bin/env python
# coding: utf-8

# In condensed matter physics, the Su − Schrieffer − Heeger-modell describes an infinte polyacetylene $(H_2C_2)_n$ chain. The Hamiltonian-operator of the model is a 2x2 Hermitian-matrix
# $$\begin{pmatrix} 
#  0 & 1+we^{-ik} \\
#  1+we^{ik} & 0 
# \end{pmatrix}
# \quad$$
# 
# The wave number $k$ can vary between  [−π,π] and  $w$ is a real number betwenn  0.5  and  1.5  
# 
# The allowed vibrational modes of the molecule are quantised according to QM. These can be determined by finding the eigenvalues of the Hamiltonian with respect to k.
# 
# The parameter $w$ can be varied manually to see how the eigenvalue distribution changes with different values

# In[9]:


get_ipython().run_line_magic('pylab', 'inline')
from ipywidgets import *

def abrazol(w):
    
    k = linspace(-pi, pi, 100) # Values for k in the allowed interval
    lambda_1 = zeros(len(k))   # Vectors for the eigenvalues (2x2 Hermitian-matrix has 2 eignevalues)
    lambda_2 = zeros(len(k))
    for i in range(len(k)):    # We calcualte the eiganvalues for each k values
        Hi = matrix([[0, 1 + w * exp(-1j * k[i])], [1 + w * exp(1j * k[i]), 0]])
    lambda_1[i], lambda_2[i] = eig(Hi)[0]
    plot(k, lambda_1)          # We plot the eigenvalues with respect to the wave number k
    plot(k, lambda_2)
    xlabel("k wave number", size=12)
    ylabel("eigenalue", size=12)
    title("Eigenvalues with respect to wave number", size=18, y = 1.05)
    xticks(linspace(-4, 4, 9))
    yticks(linspace(-3, 3, 7))

print('The eigenvalues of the Hamiltonian with respect to k, where the value of w can be varied manually')
interact(abrazol, w=FloatSlider(min=-0.5, max=1.5, step=0.02, value=0.5))


# In[ ]:




