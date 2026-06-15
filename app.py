import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
import random

  
   #PAGE CONFIG
   

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    layout="wide"
)

   
# LOAD FILES


@st.cache_resource
def load_model():
    return joblib.load("fifa2026_xgb.pkl")

@st.cache_data
def load_team_stats():
    df = joblib.load("team_stats.pkl")
    df["team"] = df["team"].replace("Turkey", "Türkiye")
    return df


@st.cache_data
def load_predictions():
    return pd.read_csv(
        "fifa2026_final_predictions.csv"
    )

xgb = load_model()
team_stats = load_team_stats()
champion_df = load_predictions()

  
# GROUPS
 

groups = {

"A": ["Mexico","South Africa","South Korea","Czech Republic"],

"B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],

"C": ["Brazil","Morocco","Haiti","Scotland"],

"D": ["United States","Paraguay","Australia","Türkiye"],

"E": ["Germany","Curaçao","Ivory Coast","Ecuador"],

"F": ["Netherlands","Japan","Sweden","Tunisia"],

"G": ["Belgium","Egypt","Iran","New Zealand"],

"H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],

"I": ["France","Senegal","Norway","Iraq"],

"J": ["Argentina","Algeria","Austria","Jordan"],

"K": ["Portugal","Uzbekistan","Colombia","DR Congo"],

"L": ["England","Croatia","Ghana","Panama"]

}
  
# HELPERS  

def get_team_stats(team):

    row = team_stats[
        team_stats["team"] == team
    ]

    if len(row) == 0:

        st.error(
            f"{team} not found in team_stats.pkl"
        )

        raise ValueError(
            f"{team} not found in team_stats.pkl"
        )

    return row.iloc[0]


def create_features(
    home_team,
    away_team
):

    home = get_team_stats(home_team)
    away = get_team_stats(away_team)

    data = {

        "home_elo": home["elo"],
        "away_elo": away["elo"],
        "elo_diff": home["elo"] - away["elo"],

        "home_form": home["form"],
        "away_form": away["form"],
        "form_diff": home["form"] - away["form"],

        "home_avg_goals": home["avg_goals"],
        "away_avg_goals": away["avg_goals"],

        "home_avg_conceded": home["avg_conceded"],
        "away_avg_conceded": away["avg_conceded"],

        "goal_attack_diff":
            home["avg_goals"]
            - away["avg_goals"],

        "goal_defense_diff":
            away["avg_conceded"]
            - home["avg_conceded"],

        "head_to_head_diff": 0,

        "home_win_rate":
            home["win_rate"],

        "away_win_rate":
            away["win_rate"],

        "neutral": 1,

        "tournament_enc": 0,

        "home_advantage":
            home["win_rate"]
            - away["win_rate"]
    }

    return pd.DataFrame([data])


  
# MATCH PREDICTION
  

def simulate_match(
    home_team,
    away_team
):

    X = create_features(
        home_team,
        away_team
    )

    probs = xgb.predict_proba(X)[0]

    outcome = np.random.choice(
        [0, 1, 2],
        p=probs
    )

    return outcome


  
# GOAL SIMULATION
  

def simulate_match_score(
    home_team,
    away_team
):

    home = get_team_stats(home_team)
    away = get_team_stats(away_team)

    home_attack = home["avg_goals"]
    away_attack = away["avg_goals"]

    home_defense = home["avg_conceded"]
    away_defense = away["avg_conceded"]

    home_xg = (
        home_attack +
        away_defense
    ) / 2

    away_xg = (
        away_attack +
        home_defense
    ) / 2

    home_xg = max(
        home_xg,
        0.3
    )

    away_xg = max(
        away_xg,
        0.3
    )

    home_goals = np.random.poisson(
        home_xg
    )

    away_goals = np.random.poisson(
        away_xg
    )

    return (
        home_goals,
        away_goals
    )


  
# GROUP STAGE
  

def initialize_group_table(
    group_teams
):

    return pd.DataFrame({

        "Team": group_teams,

        "Points": 0,
        "Wins": 0,
        "Draws": 0,
        "Losses": 0,

        "GF": 0,
        "GA": 0,
        "GD": 0

    })


def simulate_group(
    group_teams
):

    table = initialize_group_table(
        group_teams
    )

    fixtures = list(
        combinations(
            group_teams,
            2
        )
    )

    for (
        home_team,
        away_team
    ) in fixtures:

        (
            home_goals,
            away_goals
        ) = simulate_match_score(
            home_team,
            away_team
        )

        home_idx = table[
            table["Team"] == home_team
        ].index[0]

        away_idx = table[
            table["Team"] == away_team
        ].index[0]

        table.loc[
            home_idx,
            "GF"
        ] += home_goals

        table.loc[
            away_idx,
            "GF"
        ] += away_goals

        table.loc[
            home_idx,
            "GA"
        ] += away_goals

        table.loc[
            away_idx,
            "GA"
        ] += home_goals

        if home_goals > away_goals:

            table.loc[
                home_idx,
                "Points"
            ] += 3

            table.loc[
                home_idx,
                "Wins"
            ] += 1

            table.loc[
                away_idx,
                "Losses"
            ] += 1

        elif away_goals > home_goals:

            table.loc[
                away_idx,
                "Points"
            ] += 3

            table.loc[
                away_idx,
                "Wins"
            ] += 1

            table.loc[
                home_idx,
                "Losses"
            ] += 1

        else:

            table.loc[
                home_idx,
                "Points"
            ] += 1

            table.loc[
                away_idx,
                "Points"
            ] += 1

            table.loc[
                home_idx,
                "Draws"
            ] += 1

            table.loc[
                away_idx,
                "Draws"
            ] += 1

    table["GD"] = (
        table["GF"]
        - table["GA"]
    )

    table = table.sort_values(

        ["Points",
         "GD",
         "GF"],

        ascending=False

    )

    return table.reset_index(
        drop=True
    )


def simulate_all_groups():

    results = {}

    for (
        group_name,
        teams
    ) in groups.items():

        results[
            group_name
        ] = simulate_group(
            teams
        )

    return results


  
# QUALIFICATION
  

def get_32_qualified_teams(
    group_results
):

    qualified = []
    third_place = []

    for (
        group_name,
        table
    ) in group_results.items():

        table = table.sort_values(
            ["Points",
             "GD",
             "GF"],
            ascending=False
        )

        qualified.extend(
            table.head(2)["Team"]
            .tolist()
        )

        third_place.append(
            table.iloc[2]
        )

    third_df = pd.DataFrame(
        third_place
    )

    third_df = third_df.sort_values(

        ["Points",
         "GD",
         "GF"],

        ascending=False

    )

    best_third = third_df.head(8)

    qualified.extend(
        best_third["Team"]
        .tolist()
    )

    return qualified


  
# KNOCKOUTS
  

def play_knockout_match(
    team1,
    team2
):

    X = create_features(
        team1,
        team2
    )

    probs = xgb.predict_proba(
        X
    )[0]

    away_win = float(
        probs[0]
    )

    home_win = float(
        probs[2]
    )

    total = (
        home_win +
        away_win
    )

    if total <= 0:

        home_win = 0.5
        away_win = 0.5

    else:

        home_win /= total
        away_win /= total

    p = np.array([
        home_win,
        away_win
    ])

    p = p / p.sum()

    winner = np.random.choice(
        [team1, team2],
        p=p
    )

    return winner


def simulate_knockout_round(
    teams
):

    winners = []

    for i in range(
        0,
        len(teams),
        2
    ):

        winner = play_knockout_match(
            teams[i],
            teams[i + 1]
        )

        winners.append(
            winner
        )

    return winners


def simulate_world_cup(
    qualified
):

    teams = qualified.copy()

    random.shuffle(
        teams
    )

    while len(teams) > 1:

        teams = simulate_knockout_round(
            teams
        )

    return teams[0]

  
# SIDEBAR
  

st.sidebar.title(
    "⚽ FIFA 2026 Predictor"
)

page = st.sidebar.radio(

    "Navigation",

    [

        "Match Predictor",

        "Team Information",

        "World Cup Groups",

        "Tournament Simulator",

        "Monte Carlo Simulator",

        "Champion Probabilities"

    ]

)

  
# MATCH PREDICTOR
 

if page == "Match Predictor":

    st.header(
        "🔮 Match Predictor"
    )

    teams = sorted(
        team_stats["team"]
        .unique()
    )

    col1, col2 = st.columns(2)

    with col1:

        home_team = st.selectbox(
            "Team 1",
            teams
        )

    with col2:

        away_team = st.selectbox(
            "Team 2",
            teams
        )

    if st.button(
        "Predict Match"
    ):

        X = create_features(
            home_team,
            away_team
        )

        probs = xgb.predict_proba(
            X
        )[0]

        st.success(
            "Prediction Complete"
        )

        st.write(
            f"{home_team} Win: {probs[2]*100:.2f}%"
        )

        st.write(
            f"Draw: {probs[1]*100:.2f}%"
        )

        st.write(
            f"{away_team} Win: {probs[0]*100:.2f}%"
        )
        # Most Likely Score

        scores = []
        for _ in range(500):
            hg, ag = simulate_match_score(
                home_team,
                away_team
            )

            scores.append(
                (hg, ag)
            )

        predicted_score = max(
            set(scores),
            key=scores.count
        )

        st.subheader(
            "Most Likely Score"
        )

        st.success(
            f"{home_team} {predicted_score[0]} - {predicted_score[1]} {away_team}"
        )


# TEAM INFO
  

elif page == "Team Information":

    team = st.selectbox(
        "Select Team",
        sorted(
            team_stats["team"]
            .unique()
        )
    )

    stats = get_team_stats(
        team
    )

    st.metric(
        "ELO",
        round(
            stats["elo"]
        )
    )

    st.metric(
        "Form",
        round(
            stats["form"],
            2
        )
    )

    st.metric(
        "Win Rate",
        f"{stats['win_rate']*100:.2f}%"
    )

  
# GROUPS
  

elif page == "World Cup Groups":

    st.header(
        "🏆 FIFA 2026 Groups"
    )

    for group, teams in groups.items():

        st.subheader(
            f"Group {group}"
        )

        st.write(
            teams
        )

 
# TOURNAMENT
  
elif page == "Tournament Simulator":

    st.header(
        "🏆 Tournament Simulator"
    )

    if st.button(
        "Simulate Tournament"
    ):

        group_results = (
            simulate_all_groups()
        )

        st.header(
            "Group Stage Results"
        )

        for (
            group_name,
            table
        ) in group_results.items():

            st.subheader(
                f"Group {group_name}"
            )

            st.dataframe(
                table,
                use_container_width=True
            )

        qualified = (
            get_32_qualified_teams(
                group_results
            )
        )

        st.header(
            "Qualified Teams"
        )

        qualified_df = pd.DataFrame(
            {
                "Qualified Teams":
                qualified
            }
        )

        st.dataframe(
            qualified_df,
            use_container_width=True
        )

        random.shuffle(
            qualified
        )

        teams = qualified.copy()

        round_names = [

            "Round of 32",

            "Round of 16",

            "Quarter Finals",

            "Semi Finals",

            "Final"

        ]

        round_no = 0

        while len(teams) > 1:

            st.header(
                round_names[
                    round_no
                ]
            )

            winners = []

            matches = []

            for i in range(
                0,
                len(teams),
                2
            ):

                team1 = teams[i]

                team2 = teams[
                    i + 1
                ]

                home_goals, away_goals = (
                    simulate_match_score(
                        team1,
                        team2
                    )
                )

                if (
                    home_goals >
                    away_goals
                ):

                    winner = team1

                elif (
                    away_goals >
                    home_goals
                ):

                    winner = team2

                else:

                    winner = random.choice(
                        [
                            team1,
                            team2
                        ]
                    )

                matches.append(
                    {
                        "Team 1":
                        team1,

                        "Score":
                        f"{home_goals}-{away_goals}",

                        "Team 2":
                        team2,

                        "Winner":
                        winner
                    }
                )

                winners.append(
                    winner
                )

            round_df = pd.DataFrame(
                matches
            )

            st.dataframe(
                round_df,
                use_container_width=True
            )

            teams = winners

            round_no += 1

        champion = teams[0]

        st.balloons()

        st.success(
            f"🏆 FIFA World Cup Champion: {champion}"
        )

 
# MONTE CARLO
 

elif page == "Monte Carlo Simulator":

    simulations = st.slider(
        "Simulations",
        100,
        5000,
        1000,
        100
    )

    if st.button(
        "Run Monte Carlo"
    ):

        winners = {}

        progress = st.progress(
            0
        )

        for i in range(
            simulations
        ):

            group_results = (
                simulate_all_groups()
            )

            qualified = (
                get_32_qualified_teams(
                    group_results
                )
            )

            champion = (
                simulate_world_cup(
                    qualified
                )
            )

            winners[
                champion
            ] = winners.get(
                champion,
                0
            ) + 1

            progress.progress(
                (i + 1)
                /
                simulations
            )

        result_df = pd.DataFrame({

            "Team":
                winners.keys(),

            "Titles":
                winners.values()

        })

        result_df[
            "Probability"
        ] = (

            result_df["Titles"]

            /

            simulations

            *

            100

        )

        result_df = (
            result_df
            .sort_values(
                "Probability",
                ascending=False
            )
        )

        st.dataframe(
            result_df
        )

        st.bar_chart(
            result_df
            .head(10)
            .set_index(
                "Team"
            )["Probability"]
        )

  
# CHAMPION PROBABILITIES
  

elif page == "Champion Probabilities":

    st.header(
        "🏆 Champion Odds"
    )

    st.dataframe(
        champion_df
    )

    top10 = champion_df.head(
        10
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.bar(
        top10["Team"],
        top10["Probability"]
    )

    plt.xticks(
        rotation=45
    )

    st.pyplot(
        fig
    )