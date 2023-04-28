import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from vega_datasets import data
import datetime
import calendar
import plotly.express as px
import plotly.graph_objects as go
import time
import plotly.figure_factory as ff
from pycaret.regression import *
import seaborn as sns
import numpy as np

# load data


def load_data1():
    accidents_states = pd.read_csv('accidents_states_US_Accidents.csv')
    return accidents_states
@st.cache_data
def load_data3():
    df_cars_time = pd.read_pickle('whole_time_US_Accidents.pkl')
    return df_cars_time

#def load_data4():
    #df = pd.read_csv('/content/drive/MyDrive/car_accidents/US_Accidents_Dec21_updated.csv')
    #df['Visibility'] = df['Visibility(mi)']
    
    #return df



def load_data6():
    df16 = pd.read_csv('accidents_2016.csv') 
    df16['lat'] = df16['lat'].round(3)
    df16['lon'] = df16['lon'].round(3)
    df16 = df16.drop_duplicates(subset = ['lat', 'lon']) 
    return df16

def load_data7():
    df17 = pd.read_csv('accidents_2017.csv')
    df17['lat'] = df17['lat'].round(3)
    df17['lon'] = df17['lon'].round(3)
    df17 = df17.drop_duplicates(subset = ['lat', 'lon'])
    return df17

def load_data8():
    df18 = pd.read_csv('accidents_2018.csv')
    df18['lat'] = df18['lat'].round(3)
    df18['lon'] = df18['lon'].round(3)
    df18 = df18.drop_duplicates(subset = ['lat', 'lon'])
    return df18

def load_data9():
    df19 = pd.read_csv('accidents_2019.csv')
    df19['lat'] = df19['lat'].round(3)
    df19['lon'] = df19['lon'].round(3)
    df19 = df19.drop_duplicates(subset = ['lat', 'lon'])
    return df19

def load_data10():
    df20 = pd.read_csv('accidents_2020.csv')
    df20['lat'] = df20['lat'].round(3)
    df20['lon'] = df20['lon'].round(3)
    df20 = df20.drop_duplicates(subset = ['lat', 'lon'])
    return df20

def load_data11():
    df21 = pd.read_csv('accidents_2021.csv')
    df21['lat'] = df21['lat'].round(3)
    df21['lon'] = df21['lon'].round(3)
    df21 = df21.drop_duplicates(subset = ['lat', 'lon'])
    return df21

@st.cache_data
def predict_cache(test_data):
    rf_saved = load_model('rf_model')
    predictions = predict_model(rf_saved, data = test_data)
    return predictions['prediction_label']








# title



st.title('Car Accidents from 2016-2021')



with st.spinner("Please Wait..."):
  time.sleep(20)




# DATASETS




st.set_option('deprecation.showPyplotGlobalUse', False)
df_cars_time = load_data3()    # whole dataset with datetime
#df = load_data4()      # original dataset
 


df_sun = df_cars_time.groupby(['State', 'year', 'month', 'hour', 'Weather_Condition']).count()[['ID']].rename(columns = {"ID": 'num_crashes'}).reset_index()









# tab titles


tab1, tab2, tab3, tab4, tab6 = st.tabs(["Overview", "Weather PT. 1", "Weather PT. 2", "Time and Location", "Data and Resources"])





# side bar 




with st.sidebar:
    st.title("Predict your severity")

    #df_sun = df_cars_time.groupby(['State', 'year', 'month', 'hour', 'Weather_Condition']).count()[['ID']].rename(columns = {"ID": 'num_crashes'}).reset_index()

    #sunburst = px.pie(df_sun, values='num_crashes', names='State', title="Percent of Accidents by State in the US")
    #st.plotly_chart(sunburst, use_container_width=True)

    #sunburst2 = px.pie(df_sun, values='num_crashes', names='hour', title="Percent of Accidents per Hour in the US")
    #st.plotly_chart(sunburst2, use_container_width=True)

    #sunburst3 = px.pie(df_sun, values='num_crashes', names='year', title="Percent of Accidents by Year in the US")
    #st.plotly_chart(sunburst3, use_container_width=True)

    states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

    inp_percip = st.slider('Percipitation', 0.0, 0.5, 0.0, step=0.1)
    inp_temp = st.slider('Temperature', 0, 100, 50, step=1)
    inp_side = st.radio('Impact Side', ('L', 'R'), index=0)
    inp_state = st.selectbox(
      'In which state was the accident?',
      states)

    test_data = pd.DataFrame({'Precipitation(in)' : [inp_percip],
      'Temperature(F)': [inp_temp],
      'Side': [inp_side],
      'State': [inp_state]})

    st.write('Serverity of the accident on scale 1-4 = %0.0f'%predict_cache(test_data)[0])

    



# end side bar 









# tab data begin




with tab1:
  st.title("Overview")

  # overview section

  sunburst = px.pie(df_sun, values='num_crashes', names='State', title="Percent of Accidents by State in the US")
  st.plotly_chart(sunburst, use_container_width=True)
  st.write("Click the top right expansion arrows to enlarge the pie chart.")
  st.title("Tab Contents:")
  st.subheader("Sidebar:")
  st.write("Predict your Severity")
  st.subheader("Weather Pt 1:")
  st.write("Road Accident Percentage for different Weather Conditions")
  st.write("Road Accident Percentage for different Visibility range")
  st.subheader("Weather Pt 2:")
  st.write("Precipitation Vs Severity")
  st.write("Visibility vs Severity")
  st.subheader("Time and Location:")
  st.write("Top Ten States by Total Crashes")
  st.write("Bottom Ten States by Total Crashes")
  st.write("Number of Crashes vs. Time of Day")
  st.write("Number of Crashes vs. Day of Week")
  st.subheader("Data and Resource:")
  st.write("Table of States vs. Count of Crashes")
  st.write("Link to Source of Dataset")
  



  
  
  # end of overview section




with tab2:
  st.title("Weather Findings PT. 1")
  df_cars_time['Visibility'] = df_cars_time['Visibility(mi)']
  # weather first tab
  weather_condition_df = pd.DataFrame(df_cars_time.Weather_Condition.value_counts().head(10)).reset_index().rename(columns={'index':'Weather_Condition', 'Weather_Condition':'Cases'})
  weather_condition_df['Percentage'] = weather_condition_df['Cases'] / weather_condition_df['Cases'].sum() * 100

  bar = alt.Chart(weather_condition_df).mark_bar().encode(
    x='Cases:Q',
    y=alt.Y('Weather_Condition:N', sort='-x'),
    color=alt.Color('Weather_Condition'),
    tooltip=['Percentage:Q','Cases:Q']
  ).properties(height=400, title='Road Accident Percentage for different Weather Conditions').interactive()

  st.altair_chart(bar, use_container_width=True)

  st.subheader('Finding: ')
  st.write("In most of the cases (42.79%) the weather was Fair and approximately in 14% cases it was mostly cloudy.") 

  visibility_df = pd.DataFrame(df_cars_time.Visibility.value_counts().head(10)).reset_index().rename(columns={'index':'Visibility', 'Visibility':'Cases'})
  visibility_df['Percentage'] = visibility_df['Cases'] / visibility_df['Cases'].sum() * 100

  #st.write(visibility_df.Visibility.unique())

  st.title('Visibility')

  visibility_bar = alt.Chart(visibility_df).mark_bar().encode(
    y=alt.Y('Cases:Q', title='Accident Cases'),
    x=alt.X('Visibility:N', title='Visibility(mi)', axis = alt.Axis(labelAngle=15)),
    color=alt.Color('Visibility'),
    tooltip=['Percentage:Q', 'Visibility:N']
  ).properties(height=400, title='Road Accident Percentage for different Visibility range').interactive()

  st.altair_chart(visibility_bar, use_container_width=True)

  st.subheader('Finding: ')
  st.write("In maximum cases (82.24%) of road accident, the Visibility range is 10(mi).") 


  # end of weather first tab





with tab3:
  st.title("Weather Findings PT. 2")
  
  # weather second tab

  df_weather2 = df_cars_time.groupby('Visibility(mi)').agg({'Severity': np.average}).reset_index().head(50)
  df_weather3 = df_cars_time.groupby('Precipitation(in)').agg({'Severity': np.average}).reset_index().head(50)
  
  fig9 = px.line(df_weather2, x='Visibility(mi)', y='Severity', title="Driver Visibility vs. Severity")
  #fig9.show()
  st.plotly_chart(fig9)

  st.subheader("Findings:")
  st.write("This chart demonstrates the correlation between the Driver's Visibility compared to the Severity of the crash. The chart shows how there is a wide range of severity's at low visibility with a large spike  at around 1.5 Miles of visibility which could be due to the increase in speed that can happen at higher visibility. The chart begins to even out as visibility becomes clearer.")

  fig10 = px.line(df_weather3, x='Precipitation(in)', y='Severity', title="Precipitation vs. Severity")
  #fig10.show()
  st.plotly_chart(fig10)

  st.subheader("Findings:")
  st.write("The Chart demonstrates the relationship between Precipitation in inches with the Severity of the Crash. The chart shows how there is a clear jump in severity as soon as any precipitation begins but then gradually increases as the rain increases.")
  
  
  # end of weather second tab





with tab4: 
  st.title("Time and Location")

  accidents_states = load_data1() # accidents per state

  # time tab

    #Top10 states by num of Crashes

  fig = px.bar(accidents_states[:10], x='State', y='Count', title = "Top 10 States for Total Crashes")
  #top_ten = fig.show()
  st.plotly_chart(fig)

  # bot 10 states

  fig2 = px.bar(accidents_states[40:], x='State', y='Count', title = "Bottom 10 States for Total Crashes")
  #bot_ten = fig2.show()
  st.plotly_chart(fig2)

  st.subheader("Findings:")
  st.write("Charts demonstrate that the majority of crashes occur on an ocean bordering State with California almost doubling the second State Florida.")

  

  year_select = st.selectbox("Year Select (Controls following 3 charts)", ('2016', '2017', '2018', '2019', '2020', '2021'))

  st.header("Crashes per hour by year")

  if year_select == '2016':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2016)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    )
    )
    st.plotly_chart(time)
  elif year_select == '2017':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2017)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    )
    )
    st.plotly_chart(time)
  elif year_select == '2018':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2018)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    )
    )
    st.plotly_chart(time)
  elif year_select == '2019':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2019)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    ))
    st.plotly_chart(time)
  elif year_select == '2020':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2020)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    )
    )
    st.plotly_chart(time)
  elif year_select == '2021':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2021)
    time = px.histogram(data_year, x='hour')
    #time_plot = time.show()
    time.update_layout(
      xaxis_title_text='Hour of the Day',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
      ticktext= ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
    )
    )
    st.plotly_chart(time)

  st.subheader("Findings:")
  st.write("This demonstrates that a majority of crashes occur during'rush hours'. There is a spike in crashes between 6am-9am and 3pm-6pm. There is neglible differences between the morning and evening.")
  st.header("Crashes per day of the week")
  
  
  if year_select == '2016':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2016)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)
  elif year_select == '2017':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2017)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)
  elif year_select == '2018':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2018)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)
  elif year_select == '2019':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2019)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)
  elif year_select == '2020':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2020)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)
  elif year_select == '2021':
    data_year = df_cars_time.groupby('year')
    data_year = data_year.get_group(2021)
    day = px.histogram(data_year, x=data_year.Start_Time.dt.dayofweek )
    #day_plot = day.show()
    day.update_layout(
      xaxis_title_text='Day of Week',
      bargap=.2,
      xaxis = dict(
      tickvals = ['0', '1', '2', '3', '4', '5', '6'],
      ticktext= ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    )
    )
    st.plotly_chart(day)

  st.subheader("Findings:")
  st.write("Crashes slowly increashed but remained relatively the same throughout the work week before significantly dropping for the weekend.")

  st.header("Map of Crashes by Year")

  if year_select == '2016':
    df16 = load_data6()             # 2016
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2016)
    map_hour = px.scatter_mapbox(df16, lat="lat", lon="lon",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)
  elif year_select == '2017':
    df17 = load_data7()               # 2017
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2017)
    map_hour = px.scatter_mapbox(df17, lat="lat", lon="lon", color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)
  elif year_select == '2018':
    df18 = load_data8()             # 2018
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2018)
    map_hour = px.scatter_mapbox(df18, lat="lat", lon="lon",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)
  elif year_select == '2019':
    df19 = load_data9()              # 2019
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2019)
    map_hour = px.scatter_mapbox(df19, lat="lat", lon="lon",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)
  elif year_select == '2020':
    df20 = load_data10()            # 2020
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2020)
    map_hour = px.scatter_mapbox(df20, lat="lat", lon="lon",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)
  elif year_select == '2021':
    df21 = load_data11()           # 2021
    #data_year = df_cars_time.groupby('year')
    #data_year = data_year.get_group(2021)
    map_hour = px.scatter_mapbox(df21, lat="lat", lon="lon", color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-positron")
    #map_plot = map_hour.show()
    st.plotly_chart(map_hour)






# end of time tab





#with tab5:
  

  # predictions tab


  



  # end of predictions tab





with tab6:
  st.title("Database and Resources")
  
  # resource tab
  st.subheader("DataSet: Accidents Per State from 16-21")
  
  

  accidents_states = accidents_states.drop(['index','Unnamed: 0'], axis=1)
  
  st.dataframe(accidents_states)
  
  
   

  st.subheader("Link to DataSet on Kaggle")

  link='check out this [link](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)'
  st.markdown(link,unsafe_allow_html=True)



  # end of resource tab


   
