import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def column_nan_ratios(data):
    ''' 
    Function takes in DataFrame and displays bar graph of percentage of NaNs per column
    
    Args:
    -----
    DataFrame 
    '''
    nan_per_col_percentage=[]
    nan_per_col_percentage = data.isnull().sum().values/data.shape[0]
    col_names = data.columns
    df = pd.DataFrame({'Percentage missing': nan_per_col_percentage}, index=col_names)
    ax = df.plot.bar(rot=0,figsize=(20,4))
    plt.xticks(rotation=90)
    plt.title('Percentage missing in each column')
    plt.xlabel('Column Name')
    plt.ylabel('Percentage')
    plt.grid(color='g', linestyle='-', linewidth=.5)
    plt.show()

def bar_booktable(df):
    '''
    plots bar graph for counts of unique values in book table column
    
    '''
    
    book_table = df.groupby('book_table').count()['online_order']
    #calculating rartio of yes/no for book tables
    book_table = book_table/df.shape[0]
    
    #x,y for the text in the plot + the string for the output
    y = book_table.get_values().tolist()
    x = [x for x in range(len(y))]
    #zipping x,y cordinates and string value for yes/no online book option
    zip_x_y_str = zip(x,[y1 - .03 for y1 in y] ,y)
    
    df.groupby('book_table').count()
    #ploting graph to show ratio of online order options 
    ax = book_table.plot(kind='bar', color=['palegreen','limegreen'], grid=True, title='Book Table Yes/No - count, percentage' )
    ax.set_xlabel('Book Table Option on Zomato') 
    ax.set_ylabel('Percentage')
    #print the text(ratio) on the bar
    for x,y,s in zip_x_y_str:
        #print(x,y,s)
        #rounding the ratio to 3 decimal point
        s = round(s,3)
        #aligning text for the ratio on the bars
        ax.text(x,y,str(s), horizontalalignment='center',verticalalignment='center')
        

def bar_onlineorder(df):
    '''
    plots bar graph for counts of unique values in online order column
    
    '''
    
    online_order = df.groupby('online_order').count()['book_table']
    #calculating the ratio of yes/no for online ordering
    online_order = online_order/df.shape[0]
    
    #x,y for the text in the plot + the string for the output
    y = online_order.get_values().tolist()
    x = [x for x in range(len(y))]
    #zipping x,y cordinates and string value for yes/no online delivery option
    zip_x_y_str = zip(x,[y1 - .02 for y1 in y] ,y)
    
    ax = online_order.plot(kind='bar', color=['palegreen','limegreen'], grid=True, title='Online Order Yes/No - count, percentage' )
    ax.set_xlabel('Online Delivery Options on Zomato') 
    ax.set_ylabel('Percentage')
    #print the text(ratio) on the bar
    for x,y,s in zip_x_y_str:
        #print(x,y,s)
        s = round(s,3)
        ax.text(x,y,str(s), horizontalalignment='center',verticalalignment='center')
        
def rating_curve(df):
    '''
    plots bar graph for counts of unique values in rate column
    
    '''
    
    #return the numerical value of ratings from string
    rate = df.apply(lambda x: x.rate.replace(" ","") if type(x.rate) == str else x.rate, axis=1).value_counts().sort_index()
    #Ratio of the ratings, total number of ratios = total number rows - sum of missing values)
    rate = rate/(df.shape[0] - df['rate'].isnull().sum())
    
    #x,y position of the text(ratio) for the bar graph
    y = rate.get_values().tolist()
    x = [x for x in range(len(y))]
    zip_x_y_str = zip(x,[y1 + .0008 for y1 in y] ,y)
    
    colors_list = ['grey'] + ['orangered'] + ['gold']*5  + ['yellow']*5 + ['yellowgreen']*5 + ['limegreen']*5 + ['green']*5 + ['darkgreen']*5 +['grey']
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = rate.plot(kind='bar', color=colors_list, grid=True, title='Distribution of Ratings' )
    ax.set_xlabel('Ratings') 
    ax.set_ylabel('Percentage')
    #text on bar plots
    for x,y,s in zip_x_y_str:
        #print(x,y,s)
        s = round(s,3)
        ax.text(x,y,str(s), horizontalalignment='center',verticalalignment='center')
        
def rates_to_color_code(rating):
    '''
    returns color bin in string form 
    
    '''
    
    #print(rating)
    #print(type(rating))
    if type(rating) == str and rating != '-' and rating != 'NEW':
        #print("in if")
        rating = rating.replace(" ","").split('/')[0]
        if float(rating) <= 5 and float(rating) >= 4.5:
            return "4.5-5 dark green"
        elif float(rating) < 4.5 and  float(rating) >=4:
            return "4.0-4.4 green"
        elif float(rating) < 4 and float(rating) >= 3.5:
            return "3.5-3.9 light green"
        elif float(rating) < 3.5 and float(rating) >= 3:
            return "3.0-3.4 green yellow"
        elif float(rating) < 3 and float(rating) >= 2.5:
            return "2.5-2.9 yellow"
        elif float(rating) < 2.5 and float(rating) >= 2:
            return "2.0-2.4 yellow orange"
        elif float(rating) < 2 and float(rating) >= 1.5:
            return "1.5-1.9 red orange"
        elif float(rating) <1.5 and float(rating) >= 1:
            return "1.0-1.4 red"
        
    else:
        return rating
        
def color_bins(df):
    '''
    plots bar graph for different rating bins
    
    '''
    
    #ratios for the color bins
    #rates_to_color_code function returns the color of the rating
    rate_colors = df.apply(lambda x: rates_to_color_code(x.rate), axis=1).value_counts().sort_index()
    rate_colors = rate_colors/df.shape[0]
    
    #x,y and string(ratio)
    y = np.round(rate_colors.get_values(),4).tolist()
    x = [x for x in range(len(y))]
    zip_x_y_str = zip(x,[y1 + .008 for y1 in y] ,y)
    
    colors_list = ['grey','orangered','gold','yellow','yellowgreen','limegreen','green','darkgreen','grey']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = rate_colors.plot(kind='bar', color=colors_list, grid=True, title='Rating Color Category Bins' )
    ax.set_xlabel('Rating Bins and color Category') 
    ax.set_ylabel('Percentage')
    for x,y,s in zip_x_y_str:
        #print(x,y,s)
        s = round(s,4)
        ax.text(x,y,str(s), horizontalalignment='center',verticalalignment='center')
        
        
        
def listed_in_city(df):
    '''
    plots bar graph for counts with unique values in listed in city
    
    '''

    location_count = df['listed_in(city)'].value_counts()
            #print("30 Main Locations Listed on Zomato for quick searches:"df['listed_in(city)'].value_counts().shape[0],df['listed_in(city)'].value_counts().keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(location_count.keys()[:],location_count.values[:],color = 'green') 
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Locations') 
    ax.set_ylabel('Counts')
    plt.title('Locations and Respective Restaurant Counts')
    plt.xticks(rotation=90)
    plt.show()
    
def location(df):
    '''
    plots bar graph for counts with unique values in location column
    
    '''
    
    location_count = df['location'].value_counts()
    print("Total Locations:",df['location'].value_counts().shape[0],"\nLocations:",df['location'].value_counts().keys())
    print("\n Top 50 Locations with restaurant Counts")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(location_count.keys()[0:50],location_count.values[0:50]) 
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Locations') 
    ax.set_ylabel('Count')
    plt.title('Top 10 locations with highest number of Restaurants')
    plt.xticks(rotation=90)
    plt.show()
    
    return 
    
def rest_type_listedin(df):
    '''
    plots bar graph for counts with unique values in rest_type column
    
    '''
    
    location_count = df['listed_in(type)'].value_counts()
    #print(df['listed_in(type)'].value_counts().shape[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(location_count.keys()[0:10],location_count.values[0:10]/df.shape[0]) 
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Restaurant Types') 
    ax.set_ylabel('Percentage')
    plt.title('7 Restaurant Types listed on Restaurant \n and their percentage out of 51,717 Restautrants ')
    plt.xticks(rotation=90)
    plt.show()
    
    
def rest_mixed_type(df):
    '''
    plots bar graph for counts with unique values in rest_type column
    
    returns:
    -------
    (list) unique restaurants as listed by the restaurants
    
    '''
    #Restraunt types are comma separated i.e each restaurant can have multiple restraunt types
    #finding unique restaurant types
    unique_rest = {}
    rest_max_row = {1:0,2:0,3:0}
    for i in range(51717):
        if type(df.loc[i]['rest_type']) == str:
            rest_type =  df.loc[i]['rest_type'].replace(" ", "").lower()
            list_rest = rest_type.split(',')
        #print(len(list_rest))
            rest_max_row[len(list_rest)] += 1
            for rest_typ in list_rest:
                if rest_typ not in unique_rest:
                    unique_rest[rest_typ] = 1
                else:
                    unique_rest[rest_typ] +=1 
    rest_max_row[0] = df['rest_type'].isnull().sum()
    print("types counts: restaurant counts", rest_max_row)
    df_rest  = pd.Series(rest_max_row, index=rest_max_row.keys())
    
    ax = df_rest.plot(kind='bar', color='green', grid=True, title='Number of Restaurants types listed by Restaurants' )
    ax.set_xlabel('Restauraunts Labels') 
    ax.set_ylabel('Count')
    
    return unique_rest

def unique_rest_types(df, unique_rest):
    '''
    plots bar graph 
    
    returns:
    -------
    (list) unique restaurants as listed by the restaurants
    
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(unique_rest.keys(),np.array(list(unique_rest.values())))
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Restaurants Type') 
    ax.set_ylabel('Count')
    plt.title('Restaurant Type listed by Restaurant')
    plt.xticks(rotation=90)
    plt.show()
    
def unique_cuisines(df):

    unique_cuisines = {}
    cuisine_max_row = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0} #stores the maximum number of restaurant types mentioned in the column "rest_type"
    for i in range(51717):
        if type(df.loc[i]['cuisines']) == str:
            cuisines_type =  df.loc[i]['cuisines'].replace(" ", "").lower()
            list_cuisines = cuisines_type.split(',')
            cuisine_max_row[len(list_cuisines)] += 1
            for cuisines_typ in list_cuisines:
                if cuisines_typ not in unique_cuisines:
                    unique_cuisines[cuisines_typ] = 1
                else:
                    unique_cuisines[cuisines_typ] +=1 
    cuisine_max_row[0] = df['cuisines'].isnull().sum()
    df_c  = pd.Series(cuisine_max_row, index=cuisine_max_row.keys())
    
    ax = df_c.plot(kind='bar', color='green', grid=True, title='Cuisines Count' )
    ax.set_xlabel('Number of Cuisines listed per Restaurant') 
    ax.set_ylabel('Count')
    
    return unique_cuisines

def unique_cuisines_top30(df,unique_cuisines):
    
    cuisines_series  = pd.Series(unique_cuisines, index=unique_cuisines.keys()).sort_values(ascending = False) 
    fig, ax = plt.subplots(figsize=(4, 8))
    print("Total Number of Unique cuisines:",len(unique_cuisines.keys()))
    ax.barh(list(cuisines_series.keys())[:30],list(cuisines_series.values[:30]))
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Count') 
    ax.set_ylabel('Cuisines')
    plt.title('Restaurant Counts for top 30 served cuisines')
    #plt.xticks(fontsize = 9,rotation=90)
    plt.show()
    
def unique_cuisines_lowest30(df,unique_cuisines):
    
    cuisines_series  = pd.Series(unique_cuisines, index=unique_cuisines.keys()).sort_values(ascending = False) 
    fig, ax = plt.subplots(figsize=(4, 8))
    #print("Total Number of Unique cuisines:",len(unique_cuisines.keys()))
    ax.barh(list(cuisines_series.keys())[-30:-1],list(cuisines_series.values[-30:-1]))
    ax.grid(color='g', linestyle='-', linewidth=.5)
    ax.set_xlabel('Count') 
    ax.set_ylabel('Cuisines')
    plt.title('Restaurant counts for lowest 30 served cuisines')
    #plt.xticks(fontsize = 9,rotation=90)
    plt.show()
    