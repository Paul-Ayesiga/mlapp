import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def shorten_categories(categories,cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "other"
    return categorical_map

def clean_experience(x):
    if x == "More than 50 years":
        return 50
    elif x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor`s degree' in x:
        return 'Bachelor`s Degree'
    if 'Master`s degree' in x:
        return 'Master`s Degree'
    if 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

@st.cache_data
def load_data():
    df = pd.read_csv('survey_results_public.csv')
    df= df[['Country','EdLevel','Employment','YearsCodePro','ConvertedComp']]
    df = df.rename({"ConvertedComp" : "Salary"},axis=1)

    df = df.dropna()
    df = df[df['Employment'] == "Employed full-time"]
    df = df.drop('Employment', axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)

    df = df[df['Salary'] <= 250000]
    df = df[df['Salary'] >= 10000]
    df = df[df['Country'] != 'other']

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel'].apply(clean_education)

    return df

df = load_data()

def show_explore_page():
    st.write("Explore software Engineer Salaries")

    st.write("""### Stack Over Flow Survey 2020  """)

    data = df["Country"].value_counts()

    fig1 , ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")   #aspect ratio ensures that the pie is drawn as a circle

    st.write("""### Number of data from different countries  """)


    st.pyplot(fig1)


    # bar graph showing mean salary based on country
    st.write("""### Mean Salary Based on Country """)

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)


    # line graph showing mean salary based on experience
    st.write("""### Mean Salary Based on Experience """)

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)