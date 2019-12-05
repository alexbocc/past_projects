#!/usr/bin/env python
# coding: utf-8

# ## 2014 Ebola Outbreak Analysis

# #### Death cases for viral/bacterial outbreaks usually follow an exponential growth
# #### We analyse the 2014 Ebola outbreak in Africa country by country and examine the overall behaviour of the death numbers with respect to time

# In[8]:


get_ipython().run_line_magic('pylab', 'inline')

with open("ebola.txt") as f:
    data = list()                 # létrehozunk egy adatok táblázatot, amiben a forrásfájlhoz hasonló struktúrával elmentjük az adatokat
    for i in range(16):
        data += [list()]
    for row in f:
        if row[0] != "#":
            for i in range(len(data)):           # Mivel a forrásfájl oszlopaiban voltak az összetartozó adatok, ezeket rendezzük listába
                data[i] += [int(row.split()[i])] # Az adatokat pedig int tipusként tároljuk


plot(data[0], data[2], label="Guinea")
plot(data[0], data[5], label="Liberia")
plot(data[0], data[8], label="Nigeria")
plot(data[0], data[11], label="Sierra Leone")
plot(data[0], data[14], label="Senegal") 
xlabel("Time since outbreak [Days]", size=12)
ylabel("Number of Deaths", size=12)
title("Ebola related deaths with respect to time", size = 16, y = 1.05)
legend(loc=0)


# In[12]:


out = zeros(len(data[0]))
for i in range(len(data[0])):
    out[i] = data[2][i] + data[5][i] + data[8][i] + data[11][i] + data[14][i]
dout_dt = diff(out)/diff(data[0])
alpha = mean(dout_dt/out[1:])
A = out[-1] / exp(alpha * data[0][-1])
print('Constant coefficient:', A)
print('Exponential coefficient: ', alpha)

plot(data[0], out, "rx", label="data")
plot(data[0], A * exp(alpha * array(data[0])), label="fitted curve")
title("Fit between Ebola related deaths with time and the exp function", size=14, y=1.05)
legend(loc=0)
xlabel("Time since outbreak [Days]", size=12)
ylabel("Total number of deaths", size=12)


# In[11]:


deaths_oneyr = A*exp(alpha*365)
print('The total deaths in a year according to this model are: ',int(deaths_oneyr))


# #### The fitted curve visually represents the danger of these outbreaks. In a year, 668585 could have died without international humanitarian support. This also represents a limitation of the model to extrapolate to longer times, since it does not take into account external factors influencing the spread of the virus.

# In[ ]:




