# Fetching symbols from web incase the company does not exist in symb.json
# Download chromedriver according to your version of Chrome. The driver provided here is 89.0.4389.128

from selenium import webdriver
import json
from selenium.common.exceptions import NoSuchElementException, WebDriverException

with open('symb.json','r') as in_file:
    short_names = json.load(in_file)

def getSymbol(short_name):

    if short_name not in short_names.keys():
        query = '+'.join(short_name.split())
        url = 'https://www.google.com/search?q=' + query + '+nse'
        browser = webdriver.Chrome()
        browser.get(url)
        try:
            symbol = browser.find_element_by_xpath('//*[@id="knowledge-finance-wholepage__entity-summary"]/div/g-card-section/div/g-card-section/div[2]/div[2]/div/span[2]').text
        except NoSuchElementException:
            browser.quit()
            print('Sorry, the stock you asked for isn\'t available')        
        browser.quit()
        short_names[short_name] = symbol.upper() + '.NS'
        with open('symb.json','w') as out_file:
            json.dump(short_names,out_file,indent=4)
            
    return short_names[short_name]

def getFuture(symbol):
        query = 'moneycontrol '+'+'.join(symbol.split())
        url = 'https://www.google.com/search?q=' + query + '+nse'
        browser = webdriver.Chrome()
        browser.get(url)
        try:
            next_link = browser.find_element_by_xpath('//*[@id="rso"]/div[1]/div/div/div[1]/a').get_attribute('href')
            browser.get(next_link)
            stat = browser.find_element_by_class_name('trnd').text
            stat = stat.lower()
            # print(stat)
            if 'bearish' in stat:
                return -1
            elif 'bullish' in stat:
                return 1
            elif 'neutral' in stat:
                return 0
            else:
                return None
        except NoSuchElementException as e:
            print('Sorry, the stock you asked for isn\'t available')
        except WebDriverException as e:
            print('Looks like you interrupted the procedure. Please re-run if you want the results.')      
        finally:
            browser.quit()
