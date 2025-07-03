import streamlit as st
import plotly.express as px
import os
import pandas as pd
from collections import Counter
import ast
import pycountry

# ---------- Loading and Preparing data ----------
df = pd.read_csv('data/tmdb_movies_cleaned.csv')
min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())

# Ensure list columns are lists and handle empty lists
for col in ['genres', 'production_countries', 'spoken_languages']:
    if col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)
        df[col] = df[col].apply(lambda x: ['Unknown'] if isinstance(x, list) and len(x) == 0 else x)

# Aggregate yearly stats
yearly_data = {
    'releases': df.groupby('release_year').size().reset_index(name='num_releases'),
    'revenue': df.groupby('release_year')['revenue'].mean().reset_index(name='avg_revenue'),
    'rating': df.groupby('release_year')['vote_average'].mean().reset_index(name='avg_rating'),
    'popularity': df.groupby('release_year')['popularity'].mean().reset_index(name='avg_popularity'),
    'runtime': df.groupby('release_year')['runtime'].mean().reset_index(name='avg_runtime')
}
all_years = range(min_year, max_year + 1)
for key in yearly_data:
    yearly_data[key] = (
        yearly_data[key]
        .set_index('release_year')
        .reindex(all_years)
        .interpolate()
        .fillna(0)
        .reset_index()
    )

# Function to load CSS from a file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="CinemaScope Dashboard")

# Load external CSS file
load_css('static/style.css')

st.title("🎬 CinemaScope: Evolving Movie Trends (1990–2025)", anchor=False)

st.markdown("""
Explore how movie trends have evolved over the years based on TMDB data.
Use the slider below to filter the analysis by release year range.
""")

# --- Filters ---
st.markdown("<h2>Filters</h2>", unsafe_allow_html=True)
selected_years = st.slider(
    'Select Release Year Range:',
    min_year, max_year,
    (min_year, max_year),
    step=1
)

filtered_df = df.copy()
# Apply year filter
min_y, max_y = selected_years
filtered_df = filtered_df[(filtered_df['release_year'] >= min_y) & (filtered_df['release_year'] <= max_y)]

# Filter yearly aggregates
def filter_df(data):
    return data[(data['release_year'] >= min_y) & (data['release_year'] <= max_y)]

filtered = {k: filter_df(v) for k, v in yearly_data.items()}

# --- Key Performance Indicators (KPIs) ---
st.markdown("<h2>KPIs</h2>", unsafe_allow_html=True)
total_movies = len(filtered_df)
total_rev = filtered_df['revenue'].sum() / 1_000_000_000
avg_rating = filtered_df['vote_average'].mean()
avg_pop = filtered_df['popularity'].mean()
min_pop = filtered_df['popularity'].min()
max_pop = filtered_df['popularity'].max()
avg_runtime = filtered_df['runtime'].mean()

# Construct the HTML for all KPIs within a single flex container
kpis_html = f"""
<div class='kpi-container'>
    <div class='kpi-card'>
        <h5 class='kpi-label-h5'>Total Movies</h5>
        <h3 class='kpi-value-h3'>{total_movies:,}</h3>
    </div>
    <div class='kpi-card'>
        <h5 class='kpi-label-h5'>Total Revenue</h5>
        <h3 class='kpi-value-h3'>${total_rev:,.2f}B</h3>
    </div>
    <div class='kpi-card'>
        <h5 class='kpi-label-h5'>Avg Rating</h5>
        <h3 class='kpi-value-h3'>{avg_rating:.2f}/10</h3>
    </div>
    <div class='kpi-card'>
        <h5 class='kpi-label-h5'>Avg Popularity</h5>
        <h3 class='kpi-value-h3'>{avg_pop:.2f}</h3>
        <p style='font-size:0.9em; margin:0;'>({min_pop:.2f} – {max_pop:.2f})</p>
    </div>
    <div class='kpi-card'>
        <h5 class='kpi-label-h5'>Avg Runtime</h5>
        <h3 class='kpi-value-h3'>{avg_runtime:.0f}</h3>
        <p style='font-size:0.9em; margin:0;'>minutes</p>
    </div>
</div>
"""
st.markdown(kpis_html, unsafe_allow_html=True)

# --- Time Series Plots ---
st.markdown("<h2>Time Trends Analysis</h2>", unsafe_allow_html=True)
plots = {
    'Movie Releases Per Year': ('releases', 'num_releases', 'Number of Releases', None),
    'Average Revenue Per Year': ('revenue', 'avg_revenue', 'Average Revenue (USD)', 'green'),
    'Average Rating Over Time': ('rating', 'avg_rating', 'Average Rating', 'blue'),
    'Average Popularity Over Time': ('popularity', 'avg_popularity', 'Average Popularity', 'orange'),
    'Average Runtime Over Time': ('runtime', 'avg_runtime', 'Average Runtime (min)', 'purple')
}

col1, col2 = st.columns(2)

for i, (title, (key, ycol, ylabel, color)) in enumerate(plots.items()):
    fig = px.line(
        filtered[key], x='release_year', y=ycol,
        title=title,
        labels={'release_year': 'Year', ycol: ylabel},
        template='plotly_white'
    )
    fig.update_traces(mode='lines+markers', marker_size=6)
    if color:
        fig.update_traces(line_color=color)
    if ycol == 'avg_rating':
        fig.update_yaxes(range=[0, 10])
    fig.update_layout(xaxis_title_font_color='#555', yaxis_title_font_color='#555')

    # Display in dashboard
    (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

# --- Feature Importance Section ---
st.markdown("<h2>Top Features Driving Movie Popularity</h2>", unsafe_allow_html=True)
importances = pd.read_csv('data/feature_importances.csv', index_col=0).squeeze("columns")
importances_df = importances.sort_values(ascending=True).reset_index()
importances_df.columns = ['Feature', 'Importance']

fig_feat = px.bar(
    importances_df,
    x='Importance',
    y='Feature',
    orientation='h',
    labels={'Importance': 'Importance', 'Feature': 'Feature'},
    title="Feature Importance for Predicting Popularity",
    template="plotly_white",
    color_discrete_sequence=['royalblue']
)
fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
fig_feat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e8e8e8')
st.plotly_chart(fig_feat, use_container_width=True)


# --- Top Genres by Decade ---
st.markdown("<h2>Top Genres by Decade</h2>", unsafe_allow_html=True)
genre_year_df = filtered_df.explode('genres')
genre_year_df['decade'] = (genre_year_df['release_year'] // 10) * 10
decade_genre_counts = genre_year_df.groupby(['decade', 'genres']).size().reset_index(name='count')
top_genres_per_decade = decade_genre_counts.sort_values(['decade', 'count'], ascending=[True, False]).groupby('decade').head(3)

fig_genre_decade = px.bar(
    top_genres_per_decade,
    x='decade',
    y='count',
    color='genres',
    barmode='group',
    labels={'count': 'Number of Movies', 'decade': 'Decade', 'genres': 'Genre'},
    title='Top 3 Dominant Genres by Decade',
    template='plotly_white'
)
st.plotly_chart(fig_genre_decade, use_container_width=True)

# --- Seasonal Release Patterns ---
st.markdown("<h2>Seasonal Release Patterns</h2>", unsafe_allow_html=True)
if 'release_month' in filtered_df.columns:
    monthly_counts = filtered_df['release_month'].value_counts().sort_index()
    fig_seasonal = px.line(
        x=monthly_counts.index,
        y=monthly_counts.values,
        markers=True,
        labels={'x': 'Release Month', 'y': 'Number of Movies'},
        title='Seasonal Patterns in Movie Releases',
        template='plotly_white'
    )
    fig_seasonal.update_xaxes(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)

# --- Countrywise Movie Production Map ---
st.markdown("<h2>Countrywise Movie Production</h2>", unsafe_allow_html=True)

#If your country codes are ISO alpha-2 or alpha-3, use them directly.
#If they are full names, you may need to map them to ISO codes for the map.
def country_to_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

all_countries = filtered_df.explode('production_countries')['production_countries']
country_counts = Counter(all_countries)
country_counts_df = pd.DataFrame(country_counts.items(), columns=['Country', 'Movie Count'])
country_counts_df['iso_alpha'] = country_counts_df['Country'].apply(country_to_iso3)
country_counts_df = country_counts_df.dropna(subset=['iso_alpha'])

fig_country_map = px.choropleth(
    country_counts_df,
    locations='iso_alpha',
    color='Movie Count',
    color_continuous_scale='Reds',
    projection='natural earth',
    title='Movie Production by Country',
    labels={'Movie Count': 'Number of Movies'},
)
fig_country_map.update_geos(showcoastlines=True, coastlinecolor="LightGray", showland=True, landcolor="whitesmoke")
fig_country_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_country_map, use_container_width=True)

# --- Movie Distribution Pie Charts ---
st.markdown("<h2>Movie Distribution</h2>", unsafe_allow_html=True)
col_genre, col_lang = st.columns(2, gap="large")

with col_genre:
    st.subheader("By Genre")
    genre_counts = filtered_df.explode('genres')['genres'].value_counts()
    top_n = 8
    top_genres = genre_counts.head(top_n)
    other_count = genre_counts[top_n:].sum()
    pie_data = pd.concat([top_genres, pd.Series({'Other': other_count})])
    fig_genre_pie = px.pie(
        pie_data,
        values=pie_data.values,
        names=pie_data.index,
        title="Distribution of Movies by Genre"
    )
    st.plotly_chart(fig_genre_pie, use_container_width=True)

with col_lang:
    st.subheader("By Language")
    lang_counts = filtered_df['language_name'].value_counts()
    top_n = 8
    top_langs = lang_counts.head(top_n)
    other_lang = lang_counts[top_n:].sum()
    pie_lang = pd.concat([top_langs, pd.Series({'Other': other_lang})])
    fig_lang_pie = px.pie(
        pie_lang,
        values=pie_lang.values,
        names=pie_lang.index,
        title="Distribution of Movies by Language"
    )
    st.plotly_chart(fig_lang_pie, use_container_width=True)

# --- Top 10 Movies by Popularity ---
st.markdown("<h2>Top 10 Movies by Popularity</h2>", unsafe_allow_html=True)
top_movies = filtered_df.sort_values('popularity', ascending=False).head(10)
st.dataframe(
    top_movies[['title', 'genres', 'language_name', 'release_year', 'popularity', 'vote_average']].reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.markdown("Data Source: TMDB API")
st.markdown("Built with Streamlit and Plotly")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        © 2025 Neehanth Reddy. All rights reserved.<br>
        <a href='https://github.com/neehanthreddym/' target='_blank' style='margin-right: 20px;'>GitHub</a>
        <a href='https://www.linkedin.com/in/neehanthreddy/' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)