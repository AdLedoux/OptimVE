##packages
import csv
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    prog='Model_Dyna',
    description='Programme d\'optimisation de la recharge du flotte de voitures')

parser.add_argument('-i', '--input', 
    default='article.csv', 
    help="Nom du fichier d'entrée")

parser.add_argument('-o', '--output', 
    default='rendu.svg', 
    help="Chemin du fichier de sortie")

parser.add_argument('-p', '--pmax', 
    default=12, 
    type=int,
    help="Puissance maximale en kW (12, par défaut)")

parser.add_argument('-ih', '--horaire', 
    default=24, 
    type=int,
    help="Indice horaire (24, par défaut)")

parser.add_argument('-dt', '--temps', 
    default=1, 
    type=int,
    help="Pas de temps (1, par défault)")

parser.add_argument('-a', '--alpha', 
    default=1, 
    type=int,
    help="coefficient alpha prix (1, par défault)")

parser.add_argument('-b', '--beta', 
    default=1, 
    type=int,
    help="coefficient beta prix (1, par défault")

args = parser.parse_args()


##import des données 
file_name = args.input
file_output = args.output
path = "./data/" + file_name

print("Import des données à partir du fichier : " + file_name)
with open(path, 'r') as data:
    csvreader = csv.DictReader(data) 
    debut = [] ; fin = [] ; puissance = [] ; cycle = []
    for row in csvreader:
        debut.append( int(row["debut"]) )
        fin.append( int(row["fin"]) )
        puissance.append( int(row["puissance"]) )
        cycle.append( int(row["cycle"]) )

    N = len(debut) #nombre de voiture
    table = np.array([debut, fin, puissance, cycle], dtype = int).T # indice début | fin | puissance recharge | nombre de cycle



##constantes globales
print("Initialisation des constantes...")
H = args.horaire #indice horaire
dh = args.temps #pas de temps
Pmax = args.pmax  #puissance maximal en kW
a = args.alpha #coefficient alpha prix 
b = args.beta #coefficient beta prix


## Fonctions utiles
def egal_tuple(a,b):
    """verifie si deux np.array sont égaux """
    for x in a == b:
        if not x:
            return False
    return True



## Fonctions de traitement
print("Initialisation des fonctions ...")

def prix(d, alpha = 1, beta = 1):
    """Definition du prix dynamique"""
    return alpha*d + beta

def d(h, X):
    """Calcul de la demande totale en puissance à l'instant h"""
    demande = 0
    for i in range(N):
        if X[i] <= h and h <= X[i] + table[i,3] - 1:
            demande += table[i,2]
    return demande

def d_partiel(h, X_part):
    """Calcul de la demande totale en puissance à l'instant h à partir de donnée partielle"""
    demande = 0
    for (t,i) in X_part :
        if t <= h and h <= t + table[i,3] - 1:
            demande += table[i,2]
    return demande

def g(X, i):
    """Calcul du cout pour la batterie numéros i"""
    Tsi = X[i] ; Tend = Tsi + table[i,3] - 1
    cout = 0
    for h in range(Tsi,Tend+1):
        demande = d(h,X)
        cout += prix(demande, a, b) * table[i,2] * dh

        if demande > Pmax : # condition constructeur modélisé par un cout "infini"
            return np.infty

    return cout

def g_partiel(X_part, i):
    """Calcul du cout pour la batterie numéros i à partir de donnée partielle"""
    Tsi = X_part[i][0] ; Tend = Tsi + table[i,3] - 1
    cout = 0
    for h in range(Tsi,Tend+1):
        demande = d_partiel(h,X_part)
        cout += prix(demande, a, b) * table[i,2] * dh

        if demande > Pmax : # condition constructeur modélisé par un cout "infini"
            return np.infty

    return cout



##Fonctions pour le rendu graphique
def visu(t, i):
    v = np.zeros(H+1, dtype = int)
    for h in range(t, t + table[i,3]):
        v[h] = 1
    return table[i,2]*v



## Résolution Failou
print("\n" + "Début du traitement ..." + "\n")

print("Initialisation de la suite ...")
m = 0 #indice de la suite
X_pred = (-1)*np.ones(N, dtype = int) # T_{m-1}

X_part = np.array([(-1,i) for i in range(N)], dtype=int )  
#Initialisation T_{0}
for i in range(N):

    cout_liste = []
    for t in range(table[i,0], table[i,1] - table[i,3] + 2):
        X_part[i] = (t,i)
        cout_liste.append( (g_partiel(X_part[:i+1],i), t) )

    tri = sorted( cout_liste, key = lambda cout: cout[0])
    X_part[i] = (tri[0][1],i)

X = np.zeros(N,dtype = int)
for (t,i) in X_part:
    X[i] = t


print("Traitement ...")
while (not egal_tuple(X_pred,X)) and m < 100 : #critère de convergence
    m += 1 #sécurité
    X_pred = X
    for i in range(N) :

        cout_liste = [] #liste (cout,ti)
        for t in range(table[i,0], table[i,1] - table[i,3] + 2):
            X[i] = t
            cout_liste.append( (g(X,i), t) )

        tri = sorted( cout_liste, key = lambda cout: cout[0])
        X[i] = tri[0][1]



##Rendu graphique 
print("Début du rendu graphique...")           
abscisse = np.array([i for i in range(H+1)], dtype = int)
fig, axs = plt.subplots(N+1, sharex = True)
fig.subplots_adjust(hspace = 0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible = False)
plt.xticks(abscisse)

for i in range(N):
    axs[i].step(abscisse, visu(X[i], i), label = str(g(X,i)), where ='mid', color = 'k')
    axs[i].margins(x=0,y=0.2)
    axs[i].legend(loc = 'upper right')

p = [prix(d(h,X)) for h in abscisse]
axs[N].bar(abscisse, p)

plt.savefig(file_output)
plt.close()
print("Résultat : m = " + str(m))
print("Fichier résultat : " + file_output)











