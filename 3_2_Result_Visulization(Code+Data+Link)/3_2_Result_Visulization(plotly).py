"""
################  Part 2  ###################
#       Collecting Geo Information          #
#############################################
"""
"""
Created on Wed Oct  7 12:28:55 2020

@author: shaoqianchen
"""
#Twitter API account @chenshaoqian
consumer_key = "***************************"
consumer_secret = "***************************"
access_token = "***************************"
access_token_secret = "***************************"

# set access to user's access key and access secret  
auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_token_secret) 

def username_clean(ori_name):
    #Convert username from byte to str
    b = ori_name.replace("b'","")
    c = b[:-1]
    cleaned_name = c
    return cleaned_name
    

def get_twitter_loc(username,auth):
    #Function find a twitter user's location
    api = tweepy.API(auth) 
    screen_name = username
    geo = ''
    try:
        user = api.get_user(screen_name)
        ID = user.id_str 
        user_info = api.get_user(ID) 
        location = user_info.location 
        if location == "": 
            geo = "unknown"
        else: 
            geo = location
    #Filter out empty location and suspended account
    except:
        print("User Suspended") 
        geo = "unknown"
    return geo

def add_geo_col(dataframe):
    dataframe_with_geo=dataframe
    dataframe_with_geo["Geo"] = ""
    for i in range(dataframe.shape[0]):
        print(i)
        temp_user = dataframe.iloc[i]["username"]
        cleaned_name = username_clean(temp_user)
        geo_tag = get_twitter_loc(cleaned_name, auth)
        dataframe_with_geo["Geo"][i] = geo_tag
    return dataframe_with_geo





    	

"""
################  Part  3 ###################
#      Data Visulizatiuon Preparation       #
#############################################
"""

"""
Created on Wed Nov 18 00:43:28 2020

@author: shaoqianchen
"""
import pandas as pd
import us
df = pd.read_csv('Book1.csv')
df.head()

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

state_name = list()

for i in states:
    state = str(us.states.lookup(i))
    state_name.append(state)
    
data_tuples = list(zip(states,state_name))
df2 = pd.DataFrame(data_tuples, columns=['location','text'])
df2["count"] = 0
df["count"] = 0

for i in range(df.shape[0]):
    if df.iloc[i]["Description_num"] == 1:
        df["count"][i]=-2
    if df.iloc[i]["Description_num"] == 3:
        df["count"][i]=2
    if df.iloc[i]["Description_num"] == 5:
        df["count"][i]=-1
    if df.iloc[i]["Description_num"] == 6:
        df["count"][i]=1
        
geo = list(df["Geo"])
geo_count = list(df["count"])


for i in range(len(geo)):
    for j in range(len(states)):
        if states[j] in geo[i]:
            df2["count"][j]+=geo_count[i]
            
 
def df_tocsv(dataframe,export_path):
    #Export dataframe to csv
    dataframe.to_csv (export_path, header=True,index=False)
               
df_tocsv(df2, "heatmap.csv")







"""
################  Part 4  ###################
#     Result Visulization Using Plotly      #
#############################################
"""

"""
Created on Wed Nov 18 00:43:28 2020

@author: shaoqianchen
"""


# Get this figure: fig = py.get_figure("https://plotly.com/~schen13/126/")
# Get this figure's data: data = py.get_figure("https://plotly.com/~schen13/126/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="Plot 126", fileopt="extend")
# Get z data of first trace: z1 = py.get_figure("https://plotly.com/~schen13/126/").get_data()[0]["z"]

# Get figure documentation: https://plotly.com/python/get-requests/
# Add data documentation: https://plotly.com/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plotly.com/python/getting-started
# Find your api_key here: https://plotly.com/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('username', 'api_key')
trace1 = {
  "meta": {"columnNames": {
      "z": "F", 
      "text": "text", 
      "locations": "z"
    }}, 
  "type": "choropleth", 
  "zmax": 1, 
  "zmin": -1, 
  "zsrc": "schen13:125:49ef2c", 
  "z": ["-0.199005", "0.14925373", "-0.6467662", "0.04975124", "-0.2487562", "0.19900498", "-0.0497512", "0.04975124", \
  "-0.0995025", "0.04975124", "-0.0995025", "0.09950249", "-0.0995025", "-0.0995025", "0.04975124", "0.19900498", "0.09950249",\
   "0.04975124", "0.09950249", "-0.1492537", "0.09950249", "0.19900498", "0.09950249", "-0.199005", "-0.0497512", "-0.2487562", \
   "0.09950249", "0.19900498", "0.04975124", "-0.0995025", "-0.0995025", "-0.0995025", "0.54726368", "-0.8457711", "0.14925373", \
   "-0.6467662", "0.14925373", "0.44776119", "0.49751244", "-0.199005", "0.19900498", "0.14925373", "-0.0995025", "-0.5472637", \
   "-0.199005", "0.04975124", "-0.0497512", "-0.4975124", "0.09950249", "0.04975124", "-0.2487562"], 
  "marker": {"line": {
      "color": "blue)", 
      "width": 1
    }}, 
  "textsrc": "schen13:125:db3820", 
  "text": ["-4", "3", "-13", "1", "-5", "4", "-1", "1", "-2", "1", "-2", "2", "-2", "-2", "1", "4", "2", "1", "2",\
   "-3", "2", "4", "2", "-4", "-1", "-5", "2", "4", "1", "-2", "-2", "-2", "11", "-17", "3", "-13", "3", "9", "10", \
   "-4", "4", "3", "-2", "-11", "-4", "1", "-1", "-10", "2", "1", "-5"], 
  "colorbar": {"title": {"text": "Trump Margin %"}}, 
  "colorscale": [
    [-1, "blue"], [1, "red"], 
  "locationmode": "USA-states", 
  "locationssrc": "schen13:125:f97cca", 
  "locations": ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", \
  "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", 
  "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], 
  "autocolorscale": False ]

data = Data([trace1])
layout = {
  "geo": {
    "scope": "usa", 
    "center": {
      "lat": 36.86526315077527, 
      "lon": -92.76264457517746
    }, 
    "lakecolor": "blue", 
    "showlakes": True, 
    "projection": {
      "type": "albers usa", 
      "scale": 0.9075729082567466
    }
  }, 
  "title": {"text": "2016 Election"}, 
  "autosize": True, 
  "colorway": ["#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE", "#0DF9FF", "#F6F926", "#FF9616", "#479B55", \
  "#EEA6FB", "#DC587D", "#D626FF", "#6E899C", "#00B5F7", "#B68E00", "#C9FBE5", "#FF0092", "#22FFA7", "#E3EE9E", \
  "#86CE00", "#BC7196", "#7E7DCD", "#FC6955", "#E48F72"], 
  "colorscale": {
    "diverging": [
      [0, "#050aac"], [0.2, "#6a89f7"], [0.4, "#bebebe"], [0.6, "#dcaa84"], [0.8, "#e6915a"], [1, "#b20a1c"], 
    "sequential": [
      [0, "#fff5f0"], [0.125, "#fee0d2"], [0.25, "#fcbba1"], [0.375, "#fc9272"], [0.5, "#fb6a4a"], [0.625, "#ef3b2c"], [0.75, "#cb181d"], [0.875, "#a50f15"], [1, "#67000d"], 
    "sequentialminus": [
      [0, "#440154"], [0.1111111111111111, "#482878"], [0.2222222222222222, "#3e4989"], [0.3333333333333333, "#31688e"], \
      [0.4444444444444444, "#26828e"], [0.5555555555555556, "#1f9e89"], [0.6666666666666666, "#35b779"], [0.7777777777777778, "#6ece58"], [0.8888888888888888, "#b5de2b"], [1, "#fde725"], 
  "paper_bgcolor": "rgb(255, 255, 255)"]
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
