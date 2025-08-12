import base64
import threading
import time

import streamlit as st
import pandas as pd
from streamlit_authenticator import LoginError
from streamlit_star_rating import st_star_rating
from st_keyup import st_keyup

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

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
    position: relative;
    padding: 5px;
    transition: all 0.2s ease;
    height: 50px; /* stała wysokość karty */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.movie-card img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-radius: 4px;
}
.movie-card .title {
    text-align: center;
    margin-top: 8px;
    font-weight: bold;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.movie-card .genres {
    font-size: 0.85rem;
    overflow: hidden;
}
.movie-card .stars-container {
    margin-top: auto; /* pushes stars to bottom */
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- USTAWIENIA APLIKACJI ---
PAGE_SIZE = 20
NUM_COLS = 4

DEBOUNCE_MS = 1000
MIN_CHARS = 2

TMDB_BASE = "https://image.tmdb.org/t/p/w600_and_h900_bestv2"
IMDB_BASE = "https://www.imdb.com/title/"

# --- STRONA LOGOWANIA ---
if "authentication_status" not in st.session_state:
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

st.set_page_config(page_title="MovieForMe", page_icon="🎬")

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
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


    movies = load_movies()


    def img_to_base64(path):
        with open(path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        ext = path.split(".")[-1]
        return f"data:image/{ext};base64,{encoded}"



    # --- RATINGS MANAGEMENT ---
    if "ratings" not in st.session_state:
        st.session_state["ratings"] = {}


    def save_rating(movie_id):
        key = f"rate_{movie_id}"
        st.session_state["ratings"][movie_id] = st.session_state[key]


    def schedule_rated_ids_update():
        time.sleep(0.5)
        st.session_state["rated_ids"] = [
            int(k.split("_")[1])
            for k, v in st.session_state.items()
            if k.startswith("rate_") and v
        ]

    def manual_logout():
        keys_to_clear = [
            "authentication_status", "name", "username",
            "logout", "login", "failed_login_attempts"
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    # --- PAGE NAV ---
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

        rated_ids = [mid for mid, val in st.session_state["ratings"].items() if val is not None]
        positive_ids = {
            mid: rating
            for mid, rating in st.session_state["ratings"].items()
            if rating is not None and rating >= 3
        }
        num_positive = len(positive_ids)

        show_only_rated = st.sidebar.checkbox(f"Show only rated: ({len(rated_ids)})")

        if show_only_rated:
            filtered = movies[movies.movieId.isin(rated_ids)]
        elif len(search_value) >= MIN_CHARS:
            filtered = movies[movies.title_lower.str.contains(search_value.lower())]
        else:
            filtered = movies

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
        st.progress(num_positive / 20)

        recommend_btn = st.button(
            btn_recommend,
            disabled=num_positive < 20,
            type="primary",
            use_container_width=True
        )
        if recommend_btn and num_positive >= 20:
            st.success("Generating recommendations based on your liked movies...")
            # TODO: put recommendation system code here

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

    n_pages = max(1, -(-len(filtered) // PAGE_SIZE))

    col_prev, col_info, col_next = st.columns([1, 6, 1])
    with col_prev:
        st.button("« Prev", on_click=prev_page, disabled=st.session_state.page == 0)
    with col_next:
        st.button("Next »", on_click=next_page,
                  disabled=st.session_state.page >= n_pages - 1)

    col_info.markdown(f"**Page {st.session_state.page + 1}/{n_pages}**", unsafe_allow_html=True)

    print(f"SESSION PAGE: {st.session_state.page}")

    start = st.session_state.page * PAGE_SIZE
    page_items = filtered.iloc[start:start + PAGE_SIZE]

    # --- GRID ---
    for i, row in page_items.reset_index(drop=True).iterrows():
        col_idx = i % NUM_COLS

        if col_idx == 0:
            cols = st.columns(NUM_COLS)
        with cols[i % NUM_COLS]:
            st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)

            if row.poster_path:
                img_url = TMDB_BASE + row.poster_path
            else:
                img_url = img_to_base64("placeholder_600x900.png")

            imdb_url = IMDB_BASE + row.imdbId + "/"
            st.markdown(f'<a href="{imdb_url}" target="_blank">'
                        f'<img src="{img_url}"></a>', unsafe_allow_html=True)

            st.markdown(f'<div class="title">{row.title_year}</div>', unsafe_allow_html=True)

            st.caption(f"Avg. rating: ⭐ {round(row.vote_average, 2)}")

            st.caption(f"Genres: {row.genres}", unsafe_allow_html=True)

            key = f"rate_{row.movieId}"
            default = st.session_state["ratings"].get(row.movieId, None)

            if key not in st.session_state:
                st.session_state[key] = default

            st.markdown('<div class="stars-container">', unsafe_allow_html=True)
            st.feedback(
                "stars",
                key=key,
                on_change=save_rating,
                args=(row.movieId,)
            )
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["authentication_status"] is False:
    st.error("Nazwa użytkownika lub hasło niepoprawne")
else:
    st.warning("Proszę podać nazwę użytkownika i hasło")