#Importing the Libraries: 

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
from scipy.interpolate import griddata
import os


@st.cache_data
def load_data():
    return pd.read_csv('Salary_Data_Based_country_and_race.csv')


df = load_data()

# Check for missing values in each column
missing_values = df.isnull().sum()
# Drop rows with missing values
df = df.dropna()
# Check the shape of the cleaned dataset
rows_before, columns_before = df.shape
rows_after, columns_after = df.shape
rows_dropped = rows_before - rows_after

#Dropping 'Unnamed:0' Column:
df = df.drop(columns=['Unnamed: 0'])

#Removing Outliers: 
high_school_outliers = df[(df['Education Level'] == 'high school') & (df['Years of Experience'] > 5)]
# Remove the identified outliers from the cleaned dataset
df = df.drop(high_school_outliers.index)
# Verify if the outliers have been removed
remaining_outliers = df[(df['Education Level'] == 'high school') & (df['Years of Experience'] > 5)]

#Standardizing categorical Columns:
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.lower()
df.head(100)
# 1. Unify "Education Level"
df['Education Level'] = df['Education Level'].replace({"bachelor's degree": "bachelor's",  "master's degree": "master's"})
# 2. Fix typo in "Job Title"
df['Job Title'] = df['Job Title'].str.replace('juniour', 'junior')
# 3. Unify "Race"
df['Race'] = df['Race'].replace({'korean': 'asian', 'chinese': 'asian','african american':'black', 'australian': 'other', 'welsh': 'other'})
#Australian and welsh are nationalities not a race

# Display the unique values after unification
unique_values_after = {}
for col in categorical_cols:
    unique_values_after[col] = df[col].unique()
#unique_values_after

# Define the experience_interval function
def experience_interval(x):
    if 0 <= x < 5:
        return '0-5'
    elif 5 <= x < 10:
        return '5-10'
    elif 10 <= x < 15:
        return '10-15'
    elif 15 <= x < 20:
        return '15-20'
    elif 20 <= x <= 25:
        return '20-25'
    else:
        return 'Other'

# Streamlit: 

#1 Sidebar
st.sidebar.image("AUB_logo.png", use_column_width=True)
st.sidebar.title("Navigation") #Sidebar Title 
section = st.sidebar.radio("Go to", ["Data Display", 
                                     "Average Salary by Experience, Education, and Country", 
                                     "Choropleth Map by Gender",
                                     "3D Surface Plot"])

# Main title
st.title("Salary Data Visualizations")
# Data Display
variables = pd.DataFrame([
    ("1", "Age"), 
    ("2", "Gender"), 
    ("3", "Educational Level"), 
    ("4", "Job Title"), 
    ("5", "Years of Experience"), 
    ("6", "Salary"), 
    ("7", "Country"), 
    ("8", "Race")], columns=["Variable Index", "Variable Name"])

#For the first sidebar option: Data display, we will write what will be displayed when we click on it
if section == "Data Display":
    st.dataframe(df)
    if st.checkbox("Dataframe Variables"):
        st.write(variables.to_html(index=False), unsafe_allow_html=True)

# First Visualization setup: Average Salary by Years of Experience, Education Level and Country
elif section == "Average Salary by Experience, Education, and Country":
    st.subheader('Average Salary by Years of Experience, Education Level, and Country')
    st.write('This visualization displays the highest, median, and lowest salaries based on years of experience for individuals with a specific degree across different countries.')
    # In-page widgets
    experience_intervals = ['0-5', '5-10', '10-15', '15-20', '20-25']
    # Set unique keys for widgets to prevent caching issues
    experience_selected = st.selectbox("Select Experience Interval", experience_intervals, key="exp_interval_key")
    min_exp, max_exp = map(int, experience_selected.split('-'))
    countries = sorted(df['Country'].unique())
    education_levels = sorted([level for level in df['Education Level'].unique() if level != 'high school'])   
    country_selected = st.selectbox('Select Your Country', countries, key="country_key")
    edu_level_selected = st.selectbox('Select Your Education Level', education_levels, key="edu_level_key")

    
    # Filter data based on selections
    filtered_data = df[(df['Years of Experience'] >= min_exp) & 
                       (df['Years of Experience'] <= max_exp) &
                       (df['Country'] == country_selected) & 
                       (df['Education Level'] == edu_level_selected)]
    
    if not filtered_data.empty:
        avg_salary = filtered_data['Salary'].mean()
        st.write(f"Average Salary in **{country_selected}** for someone with **{edu_level_selected}** education and **{experience_selected} years** of experience is: **${avg_salary:,.2f}**")
    else:
        st.write(f"No data available for the selected criteria.")

    # Create the overall bar chart with dropdown
    df['Experience Interval'] = df['Years of Experience'].apply(experience_interval)
    grouped = df.groupby(['Experience Interval', 'Education Level', 'Country']).agg({'Salary': 'mean'}).reset_index()
    
    agg_data = {}
    for interval in grouped['Experience Interval'].unique():
        for edu in grouped['Education Level'].unique():
            sub_df = grouped[(grouped['Experience Interval'] == interval) & (grouped['Education Level'] == edu)]
            if not sub_df.empty:
                agg_data[(edu, interval)] = {
                    'high': sub_df[sub_df['Salary'] == sub_df['Salary'].max()]['Country'].values[0],
                    'low': sub_df[sub_df['Salary'] == sub_df['Salary'].min()]['Country'].values[0],
                    'median': sub_df.iloc[sub_df['Salary'].sub(sub_df['Salary'].median()).abs().argsort()[:1]]['Country'].values[0]
                }
    
    fig = go.Figure()
    edu_levels = [edu for edu in df['Education Level'].unique() if edu != "high school"]
    color_dict = {'high': 'green', 'median': 'blue', 'low': 'red'}

    for edu in edu_levels:
        for key in ['high', 'median', 'low']:
            y_values, country_labels = [], []
            for interval in ['0-5', '5-10', '10-15', '15-20', '20-25']:
                if (edu, interval) in agg_data and agg_data[(edu, interval)][key] != 'N/A':
                    country = agg_data[(edu, interval)][key]
                    y_values.append(grouped[(grouped['Country'] == country) & (grouped['Education Level'] == edu) & (grouped['Experience Interval'] == interval)]['Salary'].values[0])
                    country_labels.append(country)
                else:
                    y_values.append(0)
                    country_labels.append('N/A')
            fig.add_trace(go.Bar(x=['0-5', '5-10', '10-15', '15-20', '20-25'], y=y_values, text=country_labels, textposition='auto', name=f'{key.capitalize()} ({edu})', marker_color=color_dict[key], visible=(edu==edu_levels[0])))

    # Dropdown menu for the bar chart
    buttons = []
    for edu in edu_levels:
        visibility = [edu == trace['name'].split()[-1][1:-1] for trace in fig.data]
        buttons.append(dict(args=[{'visible': visibility}], label=edu, method='restyle'))

    fig.update_layout(
        xaxis_title='Years of Experience Interval',
        yaxis_title='Average Salary',
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        template="plotly_white",
        updatemenus=[dict(buttons=buttons, direction="down", x=1.1, y=1.15)]
    )
    st.subheader("Highest, Median, and Lowest Salary by Years of Experience, Education Level and Country")
    st.plotly_chart(fig)
    education_selected_value = edu_level_selected.lower()  # Convert the selected value to lowercase

    st.write("Interpretation:")
    st.write("When evaluating salary trends across various educational backgrounds, a few consistent patterns emerge across countries and experience levels. The USA invariably takes the lead in terms of median salaries across all education tiers, setting a high benchmark for compensation. Following closely are the UK and Canada, with their competitive salary structures, although there are nuanced deviations based on experience intervals. Australia presents an intriguing dynamic – it offers promising salaries for early career professionals, especially for PhD holders, but this growth momentum appears to decelerate in more advanced experience brackets. China, contrastingly, consistently lags behind in the salary race across all educational and experience spectrums. Across the board, an upward salary trajectory with accumulating experience is unmistakable. Nonetheless, the growth's velocity and the disparities in salaries—whether high, median, or low—manifest distinctively depending on the country in question. These observations underscore the complex interplay between education, experience, and geography in determining compensation structures")

# Choropleth Map by Gender
elif section == "Choropleth Map by Gender":
    # Group by Country and Gender to calculate the average salary
    country_gender_avg_salary = df.groupby(['Country', 'Gender']).agg({'Salary': 'mean'}).reset_index()

    # Dropdown widget for gender, excluding 'Other'
    gender_options = [gender for gender in df['Gender'].unique() if gender.lower() != 'other']
    gender_selected = st.selectbox("Select Gender", gender_options)

    # Filter data based on selected gender
    fig_data = country_gender_avg_salary[country_gender_avg_salary['Gender'] == gender_selected]

    # Create choropleth map
    fig3 = px.choropleth(fig_data, locations="Country", locationmode='country names',
                         color="Salary", hover_name="Country",
                         title=f'Average Salary by Country for {gender_selected}',
                         color_continuous_scale=px.colors.sequential.Plasma,
                         template='plotly_white')
    st.plotly_chart(fig3)

    # Display interpretation based on selected gender
    if gender_selected == "male":
        st.write("Male Interpretation: " )
        st.write("The visualization displays the average salary for males across different countries. Highlighted regions indicate countries with data, with color intensity representing the salary range. The USA, in yellow, has the highest average male salary nearing 123k. China, depicted in dark purple, and Australia, in lighter purple, follow closely with salaries slightly below the USA. The salary gradient ranges from 120k to 123k, indicating a relatively narrow salary variance across these countries for males.")
        # Add the interpretation for male here
    elif gender_selected == "female":
        st.write("Female Interpretation: " )
        st.write("The visualization showcases the average salary for females across selected countries. The USA, depicted in dark purple, has the highest average female salary approaching 110k. China, in yellow, and Australia, in light purple, follow, with their average female salaries being slightly lower than the USA. The salary gradient indicates a range from 106k to 110k, suggesting a modest salary difference across these countries for females.")
        # Add the interpretation for female here


# 3D Surface Plot
elif section == "3D Surface Plot":
    st.subheader('3D Surface Plot: Relationship between Age, Years of Experience, and Salary')

    # Data points for interpolation
    points = df[['Age', 'Years of Experience']].values
    values = df['Salary'].values

    # Create a grid of Age and Years of Experience values
    grid_x, grid_y = np.mgrid[df['Age'].min():df['Age'].max():100j,
                              df['Years of Experience'].min():df['Years of Experience'].max():100j]

    # Perform the 2D interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Create the 3D surface plot
    fig_smooth_surface = go.Figure(data=[go.Surface(z=grid_z, x=grid_x[:,0], y=grid_y[0,:], colorscale='Viridis', opacity=0.7)],
                                   layout=go.Layout(
                                        scene=dict(zaxis=dict(nticks=10, range=[np.min(grid_z), np.max(grid_z)]),
                                                   xaxis_title='Age',
                                                   yaxis_title='Years of Experience',
                                                   zaxis_title='Average Salary'),
                                        margin=dict(t=40, b=0, l=0, r=0),
                                        width=900,
                                        height=600,
                                        updatemenus=[dict(type='dropdown',
                                                          showactive=True,
                                                          y=1,
                                                          x=0.8,
                                                          xanchor='left',
                                                          yanchor='bottom',
                                                          pad=dict(t=45, r=10),
                                                          buttons=[dict(label='Top View',
                                                                        method='relayout',
                                                                        args=['scene.camera.up', dict(x=0, y=0, z=1)]),
                                                                  dict(label='Front View',
                                                                        method='relayout',
                                                                        args=['scene.camera.up', dict(x=0, y=1, z=0)]),
                                                                  dict(label='Side View',
                                                                        method='relayout',
                                                                        args=['scene.camera.up', dict(x=1, y=0, z=0)]),
                                                                  dict(label='Reset',
                                                                        method='relayout',
                                                                        args=['scene.camera', dict(center=dict(x=0, y=0, z=-0.5),
                                                                                                    eye=dict(x=1.25, y=1.25, z=1.25),
                                                                                                    up=dict(x=0, y=1, z=0))])])]))
    st.write('The visualization presents a 3D perspective of average salary based on years of experience and age. It shows that salary generally increases with experience, with notable peaks around mid-age and mid-experience levels. The range spans from 50k to over 200k, indicating significant variability in earning potential over ones career.')
    # Opacity slider
    opacity = st.slider('Set Opacity', min_value=0.1, max_value=1.0, value=0.7, key='opacity_slider')
    fig_smooth_surface.data[0].update(opacity=opacity)

    st.plotly_chart(fig_smooth_surface)
