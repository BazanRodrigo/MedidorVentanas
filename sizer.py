from PIL import Image
f = open("results.txt", "r")
for linea in f:
    if 'CoinAi' in linea:
    	coindata = linea
    if 'WindowAi' in linea:
    	windowdata = linea    	
f.close()

coinparini = coindata.index('(') +1
coinparfin = coindata.index(')') 
#Sacamos los datos de los parentecis
coinsize = coindata[coinparini:coinparfin]

winparini = windowdata.index('(')+1
winparfin = windowdata.index(')')
#Sacamos los datos de los parentecis
windowsize = windowdata[winparini:winparfin]

print(windowsize)
coma =(windowsize.index(','))
leftxwin = int(windowsize[:coma])#topy
coma += 1
windowsize = (windowsize[coma:])#topy)
print(windowsize)
coma =(windowsize.index(','))
topywin  = int(windowsize[:coma])#topy
coma += 1
windowsize = (windowsize[coma:])#topy)
print(windowsize)
coma =(windowsize.index(','))
widthwin  = int(windowsize[:coma])#topy
coma += 1
windowsize = (windowsize[coma:])#topy)
print(windowsize)
heightwin  = int(windowsize[:])#topy
windowsize = (windowsize[:])#topy)
print(windowsize)

print(coinsize)
coma =(coinsize.index(','))
leftxcoin = int(coinsize[:coma])#topy
coma += 1
coinsize = (coinsize[coma:])#topy)
print(coinsize)
coma =(coinsize.index(','))
topycoin  = int(coinsize[:coma])#topy
coma += 1
coinsize = (coinsize[coma:])#topy)
print(coinsize)
coma =(coinsize.index(','))
widthcoin  = int(windowsize[:coma])#topy
coma += 1
coinsize = (coinsize[coma:])#topy)
print(coinsize)
heightcoin  = int(coinsize[:])#topy
coma += 1
coinsize = (coinsize[:])#topy)
print(coinsize)


if(widthcoin<=heightcoin):
	mmperpix = widthcoin/2.8
	alto = (widthwin/mmperpix)
	ancho = (heightwin/mmperpix)
else:
	mmperpix = heightcoin/2.8
	alto = (widthwin/mmperpix)
	ancho = (heightwin/mmperpix)
print('La ventana mide ', alto,'cm', ancho, 'cm')