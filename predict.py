
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

import json

def phone_num_count(phone_nums):
    '''
    Returns the count of phone numbers in the column. If it is missing return 0 count instead of np.nan

    Args:
    ----
    phone_nums(str) - phone number in the string form '+91 2232323423\n+91 123233323' or np.nan

    returns:
    --------
    counts of phone number

    '''
    
    phones_n = 0
    # check for if not NaNs or missing
    if phone_nums == '0' or phone_nums == 0 or type(phone_nums) == float:
        return 0
    length = 0
    # checks for 8 continuos integers and increases the count by 1
    for i in range(len(phone_nums)):
        try:
            if int(phone_nums[i]):
                length += 1
                if length == 8:
                    phones_n += 1
        except:
            length = 0
    #return count, 0 if np.nan else phone counts
    return phones_n

def lower_(x):
    '''
    function takes a string and replcaes the spaces and returns a lower case string

    Args:
    -----
    x(str) - String input 

    returns:
    -------
    string with no spaces and lower case

    '''
    #Function returns by converting string to lower and removing the spaces
    if type(x) == str:
        return x.replace(" ","").lower()
    
def type_x(x,n):
    '''
    function separates the string by comma and returns first index if n  = 1 or second index if n = 2
    
    Args:
    ----
    x(str)- comma separated string
    n - 1 or 2 to return the comma separated string at index 0 or 1
    
    returns
    ------
    returns string at n-1 index from list created by comma separation
    '''
    # function separates the string by comma and returns first index of n  = 1 or second index if n = 2
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

def str_to_float(cost):
    '''
    function returns the float value for cost of two 
    
    Args:
    ----
    cost(str)- string form of cost '1,700'
    
    returns
    ------
    returns float value of cost else 500 i.e is the median of the cust_for_two if missing
    '''
    #returns string cost '1,700' as float 1700.0
    if type(cost) == str:
        cost = cost.replace(",","").replace(" ","")
        return(float(cost))
    else:
        #if np.nan, return median of cost_for_two for the population is 500
        return 500

def process(rows,listed_rest_types,listed_rest_city_loc,unique_rest,unique_cuisines,phone_minmax,c_minmax):
    
    '''
    Function takes in dataframe containing NEW restaurant rows and performs all the preprocesing performed on the train and test data
    
    Args:
    ----
    rows(DataFrame) - data frame containing rows of NEW restaurant
    listed_rest_types(list) - list of 7 unique types of restaurant as mentioned by Zomato for quick search - to form the dummies
    listed_rest_city_loc(list) - List of 30 locations/zones to fir all the restaurants for quick searches - to form the dummies
    unique_rest(list) - unique 25 restaurant types as listed by the restaurant on zomato - to form the dummies
    unique_cuisines(list) - list of 107 unique cusines - to form the dummies
    phone_minmax(list) - list containing the min and max count of phones in the column phone
    c_minmax(list) - list containing the min and max count of cost for two
    
    returns:
    --------
    processed rows containing 307 columns and all the restaurants having rate as "NEW"
    
    '''
    
    rows.drop(columns=['url', 'address','name','rate','votes','menu_item','dish_liked','reviews_list'],inplace = True, errors ='ignore')
    
    #calculating number of phones given
    rows['phone'] = rows.apply(lambda x: (phone_num_count(x.phone) - phone_minmax[0])/(phone_minmax[1] - phone_minmax[0]), axis=1)
    rows['cost_for_two'] = rows.apply(lambda x: (str_to_float(x.cost_for_two) - c_minmax[0])/(c_minmax[1] - c_minmax[0]) , axis = 1)
        
    #dummies for online order
    # 2 dummy columns created here
    for noyes in ['no','yes']:
        column_name1 = 'online_order_'+ noyes
        #online_order_yes and online_order_no
        rows[column_name1] = rows.apply(lambda x: 1 if x.online_order.lower() == noyes else 0, axis=1)
    #print(rows.columns)
    
    #dummies for book tabler
    # 2 dummy columns created here
    for noyes in ['no','yes']:
        column_name1 = 'book_table_'+ noyes
        #book_table_yes and book_table_no dummy created
        rows[column_name1] = rows.apply(lambda x: 1 if x.book_table.lower() == noyes else 0, axis=1)
    #print(rows.columns)
    
    #creating dummies for restauant listed in 7 types
    #7 dummy columns created here
    rows['listed_in_type'] = rows.apply(lambda x: x.listed_in_type.replace(" ","_").lower(), axis = 1)
    for rest_listed in listed_rest_types:
        rest_listed = rest_listed.lower().replace(" ","_") 
        column_name1 = 'listed_in_type_'+ rest_listed
        rows[column_name1] = rows.apply(lambda x: 1 if x.listed_in_type == rest_listed else 0, axis=1)
        
    #creating dummies for location listed in 30 types
    #30 dummy columns created here
    rows['listed_in_city'] = rows.apply(lambda x: x.listed_in_city.replace(" ","_").lower(), axis = 1)
    for rest_loc in listed_rest_city_loc:
        rest_loc = rest_loc.replace(" ","_").lower()
        column_name1 = 'listed_in_city_'+ rest_loc
        rows[column_name1] = rows.apply(lambda x: 1 if x.listed_in_city == rest_loc else 0, axis=1)
        
    # dropping location of the restaurant
    rows.drop(columns = ['location','listed_in_city','listed_in_type','online_order','book_table'],inplace = True, axis = 1) 
    
    #spliting rest types in rest_type separated by comma and storing it in two columns
    rows['rest_type'] = rows.apply(lambda x: lower_(x.rest_type), axis=1)
    rows['rest_type1'] = rows.apply(lambda x: type_x(x.rest_type,1), axis=1)
    rows['rest_type2'] = rows.apply(lambda x: type_x(x.rest_type,2), axis=1)
    
    #creating dummies for rest_type1 and rest_type2 listed in 25 types each
    #50 dummy columns created here
    for rest in unique_rest:
        #rest = rest.lower
        #print(rest)
        column_name1 = 'rest_type1_'+ rest
        column_name2 = 'rest_type2_' + rest
        rows[column_name1] = rows.apply(lambda x: 1 if x.rest_type1 == rest else 0, axis=1)
        rows[column_name2] = rows.apply(lambda x: 1 if x.rest_type2 == rest else 0, axis=1)
    
    #dropping columns after creting dummies
    rows.drop(columns =['rest_type1','rest_type2','rest_type'],inplace = True) 
    
    #spliting first two cuisines in cuisines_1 and cuisines_2 separated by comma
    rows['cuisines'] = rows.apply(lambda x: lower_(x.cuisines), axis=1)
    rows['cuisines_1'] = rows.apply(lambda x: type_x(x.cuisines,1), axis=1)
    rows['cuisines_2'] = rows.apply(lambda x: type_x(x.cuisines,2), axis=1)
    
    
    #creating dummies for cuisines_1 and cuisines_2 listed in 107 types each
    #214 dummy columns created here
    for cuisine in unique_cuisines:
        #cuisine = cuisine.lower()
        column_name1 = 'cuisines_1_'+ cuisine
        column_name2 = 'cuisines_2_' + cuisine
        rows[column_name1] = rows.apply(lambda x: 1 if x.cuisines_1 == cuisine else 0, axis=1)
        rows[column_name2] = rows.apply(lambda x: 1 if x.cuisines_2 == cuisine else 0, axis=1)
    
    #dropping columns after creating dumies
    rows.drop(columns =['cuisines_1','cuisines_2','cuisines'],inplace = True)
    
    return rows         

def predict(model,X):
    '''
    Prints graph for top three predictions with probabilities
    
    args:
    ----
    model(NN) : Trained neural network
    X - processed row to be predicted
    '''
   
    
    #freeze the gradients of the model
    with torch.no_grad():
        #remove the dropouts
        model.eval()
        
        #predict
        prediction = model(X)
        ps = torch.exp(prediction)
        
        #get top 3 prediction
        top_p, top_class = ps.topk(3)
        top_p = top_p.cpu().numpy().tolist()[0]
        top_class = top_class.cpu().numpy().tolist()[0]
       
        #print(top_class)
        with open('rates.json', 'r') as f:
            class_to_bin = json.load(f)
        
        #convert the prediction class to bins in string
        list_str = [class_to_bin.get(str(int(x))) for x in top_class]
         
        #print(list_str,top_p)
        
        #plot the probabilties and predicted top bins 
        plt.barh(list_str,top_p)
        plt.xlabel("probability")
        plt.ylabel("Prediction class")
        plt.title("Prediction and probabilities")
        #print("The Bin Predicted by the model is",list_str[0],"The probability of predicting based on training data is:",top_p[0])
        
        