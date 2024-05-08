#the code opens the file and print it
import pandas as pd
users=pd.read_excel('Medicion 01 -12-2023.xlsx')
#print(users)
datos = users.iloc[:, 1:-1]
#print(datos)
data = datos.dropna(thresh=len(users.columns)-1)
print (data)

#graficar datos
import matplotlib.pyplot as plt
'''
for column in users.columns:
    plt.plot(users[column], label=column)

plt.legend()
plt.show()
'''
