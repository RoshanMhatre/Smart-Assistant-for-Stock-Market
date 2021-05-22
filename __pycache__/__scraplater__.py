from nsetools import Nse
import json
nse = Nse()
all_stock_codes = nse.get_stock_codes()

xdata = {}
for symb, comp in all_stock_codes.items():
    xdata[comp] = symb 

with open("Symbols.json", "w") as outfile: 
    json.dump(all_stock_codes, outfile,indent=4)

