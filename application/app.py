import base64
import threading
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from streamlit_authenticator import LoginError
# from streamlit_star_rating import st_star_rating
from st_keyup import st_keyup

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os

import recommend
from two_tower_model import generate_recommendation

import implicit

st.markdown("""
<style>
div[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.sidebar-header {
    flex: 0 0 auto;
    text-align: center;
    padding: 1rem;
    border-bottom: 1px solid #ccc;
}
.sidebar-content {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 0.5rem 1rem;
}
.movie-card {
    background: #262730;
    position: relative;
    padding: 5px;
    margin-bottom: 20px;
    border-radius: 15px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 600px;
    max-width: 300px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.movie-card:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.movie-card img {
    width: 100%;
    height: 400px;
    object-fit: cover;
    border-radius: 15px;
    margin-bottom: 8px;
}
.title {
    text-align: center;
    margin-top: 8px;
    font-weight: bold;
    font-size: 1.2rem; 
    margin-bottom: 8px;
}
.genres {
    font-size: 0.85rem;
    overflow: hidden;
    margin-bottom: 5x;
}
.stars-container {
    margin-top: auto;
    text-align: center;
    padding-top: 8px;
    padding-bottom: 5px;
}
.rated-on {
    text-align: center;
    font-size: 0.9rem;
    color: #bbb;
    margin-bottom: 0.5rem;
}
div[role="radiogroup"] {
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- USTAWIENIA APLIKACJI ---
PAGE_SIZE = 30
NUM_COLS = 5

DEBOUNCE_MS = 1000
MIN_CHARS = 2

TMDB_BASE = "https://image.tmdb.org/t/p/w600_and_h900_bestv2"
IMDB_BASE = "https://www.imdb.com/title/"

USE_CF = True

# --- USTAWIENIA STRONY ---
st.set_page_config(page_title="MovieForMe", page_icon="🎬")

def img_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    ext = path.split(".")[-1]
    return f"data:image/{ext};base64,{encoded}"

def set_background(path):
    encoded_bg = img_to_base64(path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{encoded_bg}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- STRONA LOGOWANIA ---
if "authentication_status" not in st.session_state:
    st.markdown(
            f"""
            <div class="sidebar-header">
                <img src="{img_to_base64("logo.png")}"
                    alt="Moje Logo"
                    style="width:40%; height:auto; object-fit:contain;">
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.session_state["authentication_status"] = None
    st.session_state["name"] = ""
    st.session_state["username"] = ""

with open("auth_config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    st.set_page_config(layout="wide")

    @st.cache_data
    def load_movies():
        df = pd.read_csv("Movies_final_ML.csv")  # WCZYTANIE DANYCH DO ZMIANY Z BAZA
        df["poster_path"] = df.poster_path.fillna("")
        df["release_date"] = pd.to_datetime(df.release_date, errors="coerce")
        df["year"] = df.release_date.dt.year.astype("Int64")
        df["title_year"] = df.apply(lambda r: f"{r.title} ({r.year})" if pd.notna(r.year) else r.title, axis=1)
        df["title_lower"] = df["title"].str.lower()
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        return df[["movieId", "imdbId", "title_year", "title_lower", "poster_path", "vote_average", "genres", "popularity"]]

    set_background("login_bg_cut.png")
    movies = load_movies()

    # --- RATINGS MANAGEMENT ---
    if "ratings" not in st.session_state:
        st.session_state["ratings"] = {}


    def save_rating(movie_id):
        key = f"rate_{movie_id}"
        rating = st.session_state[key]

        if rating is None:
            st.session_state["ratings"].pop(movie_id, None)
        else:
            st.session_state["ratings"][movie_id] = {
                "value": rating,
                "timestamp": datetime.now()
            }


    def schedule_rated_ids_update():
        time.sleep(0.5)
        st.session_state["rated_ids"] = [
            int(k.split("_")[1])
            for k, v in st.session_state.items()
            if k.startswith("rate_") and v
        ]

    # --- PAGE NAV ---
    def manual_logout():
        keys_to_clear = [
            "authentication_status", "name", "username",
            "logout", "login", "failed_login_attempts"
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    def reset_page():
        st.session_state.page = 0


    def clamp_page():
        st.session_state.page = max(0, min(st.session_state.page, n_pages - 1))


    def prev_page():
        st.session_state.page -= 1
        clamp_page()


    def next_page():
        st.session_state.page += 1
        clamp_page()


    # --- SIDEBAR ---
    with st.sidebar:

        st.markdown(
            f"""
            <div class="sidebar-header">
              <img src="{img_to_base64("logo.png")}"
                   alt="Moje Logo"
                   style="width:100%; height:auto; object-fit:contain;">
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        prev_mode = st.session_state.get("last_mode", None)
        prev_search = st.session_state.get("last_search", "")

        search_value = st_keyup(
            label="Search for title…",
            placeholder="Star Wars",
            key="live_search",
            debounce=DEBOUNCE_MS,  # <2>
            value="",  # domyślna wartość
        )

        if any(k.startswith("rate_") for k in st.session_state):
            st.session_state["last_rated_update"] = time.time()
            threading.Thread(target=schedule_rated_ids_update, daemon=True).start()

        rated_ids = [
            mid for mid, rating in st.session_state["ratings"].items()
            if isinstance(rating, dict) and rating.get("value") is not None
        ]
        positive_ids = {
            mid: rating
            for mid, rating in st.session_state["ratings"].items()
            if isinstance(rating, dict) and (rating.get("value") or 0) >= 3
        }
        num_positive = len(positive_ids)

        show_only_rated = st.sidebar.checkbox(f"Show only rated: ({len(rated_ids)})")

        if (prev_mode != show_only_rated) or (prev_search != search_value):
            if 'recommendations' in st.session_state:
                del st.session_state['recommendations']
            reset_page()

        if show_only_rated:
            filtered = movies[movies.movieId.isin(rated_ids)].copy()

            filtered["rating_time"] = filtered["movieId"].apply(
                lambda mid: st.session_state["ratings"][mid]["timestamp"]
            )

            filtered = filtered.sort_values(by="rating_time", ascending=False)

        elif len(search_value) >= MIN_CHARS:
            filtered = movies[movies.title_lower.str.contains(search_value.lower())]
        else:
            filtered = movies

        if not show_only_rated:
            filtered = filtered.sort_values(
                by="popularity",
                ascending=False,
                na_position="last"
            )

        if (prev_mode != show_only_rated) or (prev_search != search_value):
            reset_page()

        st.session_state.last_mode = show_only_rated
        st.session_state.last_search = search_value
        st.markdown('</div>', unsafe_allow_html=True)

        btn_recommend = f"Get Recommendations ({num_positive}/{20})"
        if num_positive > 19:
            st.success("You can check your recommendations!")
        else:
            st.progress(num_positive / 20)

        recommend_btn = st.button(
            btn_recommend,
            disabled=num_positive < 20,
            type="primary",
            use_container_width=True
        )
        if recommend_btn and num_positive >= 20:
            with st.spinner("Generating recommendations..."):
                BASE_DIR = Path(os.getcwd()).parent
                DATA_DIR = BASE_DIR / 'data'

                df_users = pd.read_parquet(DATA_DIR / 'user_features_clean_warm.parquet')
                df_movies = pd.read_csv(DATA_DIR / 'Movies_final_ML.csv')
                df_LOOCV = pd.read_parquet(DATA_DIR / 'ratings_LOOCV.parquet')
                df_ratings = pd.read_parquet(DATA_DIR / 'ratings_groupped_20pos.parquet')

                movieId_to_idx = generate_recommendation.get_movies_idx(df_users, df_ratings, df_LOOCV)
                idx_to_movieId = {v: k for k, v in movieId_to_idx.items()}

                if not USE_CF:
                    user_tower, device = generate_recommendation.get_user_tower('two_tower_model/user_tower.pth')

                    # 1. Prepare the user's feature row
                    # The `recommend` module now contains our new function
                    u_row = recommend.prepare_new_user_features(st.session_state['ratings'], movies)

                    seen_movies = []
                    for movieId, _ in st.session_state['ratings'].items():
                        seen_movies.append(movieId)
                    print(f"Seen movies:{seen_movies}")

                    recommendations = generate_recommendation.generate_user_emb_and_find_recommendations(df_movies,movieId_to_idx,user_tower, device,u_row, seen_movies)
                    print(recommendations)

                    st.session_state['recommendations'] = recommendations
                else:
                    ratings = st.session_state['ratings']

                    data = recommend.prepare_collaborative_filtering_data(ratings, movieId_to_idx)

                    als_model = implicit.als.AlternatingLeastSquares().load('../CF_model/collaborative_filtering.npz')

                    recommendations, scores = als_model.recommend(
                        userid=0,
                        user_items=data,
                        N=20,
                        filter_already_liked_items=True
                    )

                    recommendations = [{'movieId': idx_to_movieId[id]} for id in recommendations]
                    print(recommendations)

                    st.session_state['recommendations'] = recommendations
                    

                st.rerun()

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        st.markdown(f"Logged in as: **{st.session_state.get('name')}**", unsafe_allow_html=True)
        authenticator.logout("Logout", "sidebar", key="unique_logout_key")

    # --- MAIN PAGE ---
    if "page" not in st.session_state:
        reset_page()

    # --- RECOMMENDATIONS PAGE ---
    if 'recommendations' in st.session_state and st.session_state['recommendations']:
        st.header("Your Top Recommendations")

        recommended_ids = st.session_state['recommendations']
        recommended_ids_ordered = [rec['movieId'] for rec in recommended_ids]

        recs_df = movies[movies['movieId'].isin(recommended_ids_ordered)].copy()

        # Order of recommendations
        recs_df['movieId'] = pd.Categorical(recs_df['movieId'], categories=recommended_ids_ordered, ordered=True)
        recs_df = recs_df.sort_values('movieId')

        top_4_recs = recs_df.head(5)
        remaining_recs = recs_df.iloc[5:]

        cols = st.columns(5)
        for i, (_, row) in enumerate(top_4_recs.iterrows()):
            with cols[i]:
                if row.poster_path:
                    img_url = TMDB_BASE + row.poster_path
                else:
                    img_url = img_to_base64("placeholder_600x900.png")
                imdb_url = IMDB_BASE + row.imdbId + "/"

                st.markdown(f"""
                    <h1>{i+1}.</h1>
                       <div class="movie-card">
                           <a href="{imdb_url}" target="_blank">
                               <img src="{img_url}">
                           </a>
                           <div class="title">{row.title_year}</div>
                           <div class="genres">Avg. rating: ⭐ {round(row.vote_average, 2)}</div>
                           <div class="genres">Genres: {row.genres}</div>
                       </div>
                       """, unsafe_allow_html=True)

        if not remaining_recs.empty:
            with st.expander("See more recommendations..."):
                for _, row in remaining_recs.iterrows():
                    st.write(f"- {row['title_year']}")

        if st.button("Back to Browsing"):
            del st.session_state['recommendations']
            st.rerun()

    else:
        # --- STANDARD APP PAGE ---
        n_pages = max(1, -(-len(filtered) // PAGE_SIZE))
        start = st.session_state.page * PAGE_SIZE
        page_items = filtered.iloc[start:start + PAGE_SIZE]

        col_prev, col_info, col_next = st.columns([1, 12, 1])
        with col_prev:
            st.button("« Prev", on_click=prev_page, disabled=st.session_state.page == 0, use_container_width=True)
        with col_info:
            st.markdown(
                "<div style='text-align:center;font-weight:bold;font-size: 1.2rem;'>"
                f"Page {st.session_state.page + 1}/{n_pages}"
                "</div>",
                unsafe_allow_html=True
            )
            st.write("")
            st.write("")
            st.write("")

            # --- GRID ---
            for i, row in page_items.reset_index(drop=True).iterrows():
                col_idx = i % NUM_COLS

                if col_idx == 0:
                    cols = st.columns(NUM_COLS)

                with cols[i % NUM_COLS]:
                    if row.poster_path:
                        img_url = TMDB_BASE + row.poster_path
                    else:
                        img_url = img_to_base64("placeholder_600x900.png")

                    imdb_url = IMDB_BASE + row.imdbId + "/"

                    rating = st.session_state["ratings"].get(row.movieId)
                    if rating and "timestamp" in rating:
                        ts = rating["timestamp"].strftime("%Y-%m-%d %H:%M")
                        rated_html = f'<div class="rated-on">Rated on: {ts}</div>'
                    else:
                        rated_html = ""

                    st.markdown(f"""
                        <div class="movie-card">
                            <a>{rated_html}</a>
                                <a href="{imdb_url}" target="_blank">
                                    <img src="{img_url}">
                                </a>
                            <div class="title">{row.title_year}</div>
                            <div class="genres">Avg. rating: ⭐ {round(row.vote_average, 2)}</div>
                            <div class="genres">Genres: {row.genres}</div>
                            <div class="stars-container">
                        """, unsafe_allow_html=True)

                    key = f"rate_{row.movieId}"
                    rating_data = st.session_state["ratings"].get(row.movieId, None)

                    default_value = None
                    if isinstance(rating_data, dict):
                        default_value = rating_data.get("value")

                    if key not in st.session_state:
                        st.session_state[key] = default_value

                    col_l, col_c, col_r = st.columns([1, 3, 1])
                    with col_c:
                        st.feedback(
                            "stars",
                            key=key,
                            on_change=save_rating,
                            args=(row.movieId,),
                        )

                    st.markdown("</div></div>", unsafe_allow_html=True)

        with col_next:
            st.button("Next »", on_click=next_page,disabled=st.session_state.page >= n_pages - 1, use_container_width=True)

        print(f"SESSION PAGE: {st.session_state.page}")

        start = st.session_state.page * PAGE_SIZE
        page_items = filtered.iloc[start:start + PAGE_SIZE]


elif st.session_state["authentication_status"] is False:
    st.set_page_config(layout="centered")
    set_background("main_bg.png")
    st.error("Nazwa użytkownika lub hasło niepoprawne")
else:
    st.set_page_config(layout="centered")
    set_background("main_bg.png")
    st.warning("Proszę podać nazwę użytkownika i hasło")