import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
def rates(rating):
    '''
    Takes string form of rating and returns the float type example '3.4 /5' returns 3.4
    '''
    
    if type(rating) == str:
        rating = rating.replace(" ","").split('/')[0]
        return float(rating)
    else:
        return rating

def phone_num_count(phone_nums):
    '''
    counts the phone number from string and returns count
    example "+91 231231231\n +91312312312" returns 2
    the function returns 0 if missing
    '''
    phones_n = 0
    if phone_nums == '0' or phone_nums == 0 or type(phone_nums) == float:
        return 0
    length = 0
    for i in range(len(phone_nums)):
        try:
            if int(phone_nums[i]):
                length += 1
                if length == 8:
                    phones_n += 1
        except:
            length = 0
    return phones_n

def str_to_float(cost):
    '''
    Retruns the float from the string cost for two
    example '1,700' returns 1700 
    '''
    
    if type(cost) == str:
        cost = cost.replace(",","").replace(" ","")
        return(float(cost))
    else:
        return np.nan
    
def lower_(x):
    '''
    returns the lower case of string after removing all the space 
    example 'CAsual Dining' returns 'casualdining'
    '''
    if type(x) == str:
        return x.replace(" ","").lower()
    
def type_x(x,n):
    '''
    splits string after replacing spaces and converting it to lower case by , and returns the n-1 index string 
    example ('Cafe, Dining',1) will return cafe and ('Cafe, Dining',2) returns dining
    returns np.nan if missing
    
    '''
    if type(x) == str:
        #x = x.replace(" ","")
        x = x.replace(" ","").lower()
        x = x.split(",")
        if len(x) >= n:
            return(x[n-1])
        else:
             return np.nan
    else:
        return np.nan
    
    
    
def target_rates_to_color(rating):
    '''
    function returns labels for raitngs
    '''
    
    if rating <= 5 and rating >= 4.5:
        return 7
    elif rating < 4.5 and  rating >=4:
        return 6
    elif rating < 4 and rating >= 3.5:
        return 5
    elif rating < 3.5 and rating >= 3:
        return 4
    elif rating < 3 and rating >= 2.5:
        return 3
    elif rating < 2.5 and rating >= 2:
        return 2
    elif rating < 2 and rating >= 1.5:
        return 1
    elif rating <1.5 and rating >= 1:
        return 0
    
            

 