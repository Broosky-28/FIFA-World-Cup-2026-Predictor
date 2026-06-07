# ⚽ FIFA World Cup 2026 Predictor

## Predicting the Future of International Football with Machine Learning

Football is one of the most unpredictable sports in the world. Every World Cup sparks debates among fans, analysts, and experts about which nation will lift the trophy.

This project was built to answer a simple but fascinating question:

**"Who is most likely to win the FIFA World Cup 2026?"**

Instead of relying on opinions, FIFA rankings, or media predictions, this project uses **Machine Learning, Football Analytics, Elo Ratings, and Monte Carlo Simulation** to estimate match outcomes and tournament-winning probabilities.

The system predicts:

• Match Win / Draw / Loss Probabilities
• Group Stage Outcomes
• Knockout Stage Results
• FIFA World Cup Champions
• Championship Probabilities through 10,000 Tournament Simulations

   

# 🎯 Why I Built This Project

As both a football enthusiast and a machine learning student, I wanted to build a project that combines sports analytics with predictive modeling.

Football matches are influenced by many factors:

• Team Strength
• Current Form
• Goal Scoring Ability
• Defensive Stability
• Historical Performance
• Tournament Pressure

Traditional rankings often fail to capture these dynamics.

The goal of this project was to create a data-driven football prediction engine capable of forecasting both individual matches and entire tournaments using real historical data and advanced simulation techniques.

This project demonstrates how Machine Learning can be applied to a real-world sports forecasting problem.

   

# 🚀 Features

## ⚽ Match Outcome Simulator

One of the main features of the application is an interactive match simulator.

Users can select any two international football teams and instantly receive:

• Team 1 Win Probability
• Draw Probability
• Team 2 Win Probability

The prediction is generated using a trained XGBoost model and advanced team performance metrics.

Example:

Argentina vs France

Argentina Win: 42%
Draw: 27%
France Win: 31%

This allows users to explore hypothetical matchups and compare national teams beyond FIFA rankings.

   

## 📊 Team Analytics

The application maintains statistical profiles for every team.

Metrics include:

• Elo Rating
• Current Form
• Average Goals Scored
• Average Goals Conceded
• Win Rate

These statistics are updated and used as inputs for prediction models.

   

## 🏆 FIFA World Cup Tournament Simulator

The complete FIFA World Cup tournament structure is simulated.

Tournament stages include:

• Group Stage
• Round of 32
• Round of 16
• Quarter Finals
• Semi Finals
• Final

The simulator automatically progresses teams through the tournament and determines a champion.

        

## 🎲 Monte Carlo Simulation

A single tournament simulation does not tell the full story.

Therefore, the entire FIFA World Cup is simulated:

**10,000 Times**

Each simulation:

1. Plays every group-stage match
2. Determines qualified teams
3. Simulates knockout rounds
4. Records the champion

The championship frequency is then converted into a winning probability.

       

## 🌐 Interactive Streamlit Dashboard

The project includes an interactive web application built using Streamlit.

Dashboard Features:

• Match Outcome Simulator
• Team Statistics Viewer
• Championship Probability Rankings
• Tournament Simulator
• Interactive Visualizations

        

# 📊 Dataset

This project uses historical international football match data.

Primary Dataset:

[International Football Results (1872–2017) Dataset on Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2    17?utm_source=chatgpt.com)

The dataset contains:

• International football matches dating back to 1872
• Match results and scores
• Tournament information
• Home and away teams
• Historical football records

From this data, additional football analytics features were engineered.

       

# 🧠 Feature Engineering

Raw football data was transformed into predictive features.

Features used in the model include:

• Home Elo Rating
• Away Elo Rating
• Elo Difference

• Home Form
• Away Form
• Form Difference

• Home Average Goals Scored
• Away Average Goals Scored

• Home Average Goals Conceded
• Away Average Goals Conceded

• Goal Attack Difference
• Goal Defense Difference

• Head-to-Head Difference

• Home Win Rate
• Away Win Rate

• Neutral Venue Indicator

• Tournament Encoding

• Home Advantage

These features help capture both long-term team quality and short-term performance trends.

       

# ⚽ Football Metrics Used

## Elo Rating

Elo ratings are widely used to evaluate team strength.

Higher Elo ratings generally indicate stronger teams.

       

## Current Form

Recent performances are often more important than older matches.

Current form captures momentum entering a tournament.

    

## Goal Statistics

Each team's:

• Average Goals Scored
• Average Goals Conceded

are used to estimate attacking and defensive strength.

   

## Win Rate

Represents consistency across recent matches.

   

## Home Advantage

Captures performance differences between favorable and neutral playing conditions.

   

# 🤖 Why XGBoost?

Several machine learning models were considered during development:

• Logistic Regression
• Decision Tree
• Random Forest
• XGBoost

After experimentation, XGBoost produced the strongest overall results.

### Why XGBoost Was Selected

### 1. Higher Predictive Accuracy

XGBoost consistently achieved better performance than simpler models.

   

### 2. Handles Complex Relationships

Football outcomes depend on many interacting variables.

Examples:

• Elo Difference
• Team Form
• Goal Statistics

These relationships are often non-linear.

XGBoost effectively captures these patterns.

   

### 3. Excellent Performance on Tabular Data

Football match datasets are structured tabular datasets.

XGBoost is one of the most effective algorithms for this type of data.

   

### 4. Fast Prediction Speed

Tournament simulations require thousands of predictions.

XGBoost is efficient enough to support large-scale Monte Carlo simulations.

   

### 5. Interpretability

Feature importance can be analyzed to understand which factors most influence predictions.

   

# 🏆 World Cup Simulation Methodology

## Group Stage

Each group-stage match is simulated using:

• Team Attacking Strength
• Team Defensive Strength

Goals are generated using a Poisson distribution model.

The simulator tracks:

• Points
• Wins
• Draws
• Losses
• Goals For
• Goals Against
• Goal Difference

Teams are ranked using:

1. Points
2. Goal Difference
3. Goals Scored

   

## Qualification Process

From each group:

• Top Two Teams Qualify Automatically

Additionally:

• Best Third-Placed Teams Also Qualify

This mirrors the expanded FIFA World Cup format.

   

## Knockout Stage

For knockout matches:

• Draw probabilities are removed
• Win probabilities are normalized
• A winner is selected probabilistically

Knockout rounds include:

• Round of 32
• Round of 16
• Quarter Finals
• Semi Finals
• Final

   

# 📈 Example Outputs

The system generates:

• Match Predictions
• Group Tables
• Qualified Teams
• Tournament Brackets
• Championship Probabilities

Example:

Argentina — 18.4%
France — 16.7%
Brazil — 14.2%
England — 11.8%
Portugal — 8.3%

Actual values depend on simulation results and model updates.

   

# 📂 Project Structure

FIFA2026_Project

• app.py

• requirements.txt

• README.md

• fifa2026_xgb.pkl

• team_stats.pkl

• fifa2026_final_predictions.csv

• .gitignore



# ⚙️ Installation

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/FIFA2026-Predictor.git](https://github.com/YOUR_USERNAME/FIFA2026-Predictor.git)

Move into the project directory:

cd FIFA2026-Predictor

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.p

# 🛠 Technologies Used

• Python

• Pandas

• NumPy

• Scikit-Learn

• XGBoost

• Streamlit

• Matplotlib

• Joblib

 

# 🔮 Future Improvements

Planned upgrades include:

• Live FIFA Rankings Integration

• Dynamic Elo Updates

• Expected Goals (xG) Models

• Player-Level Statistics

• Injury and Suspension Tracking

• Interactive Tournament Brackets

• Real-Time Match Updates

• Automated Data Refresh Pipelines

   

# 👨‍💻 About the Developer

I built this project to combine my interests in:

• Football Analytics

• Machine Learning

• Data Science

• Predictive Modeling

This project demonstrates how historical sports data can be transformed into intelligent prediction systems through feature engineering, machine learning, probabilistic modeling, and large-scale simulation.

Beyond predicting match outcomes, the project serves as a practical example of applying AI and data science techniques to a real-world forecasting challenge.

    

# 🙏 Acknowledgements

Special thanks to:

• [Kaggle](https://www.kaggle.com?utm_source=chatgpt.com)

• [International Football Results Dataset by Mart Jürisoo](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?utm_source=chatgpt.com)

• The Open-Source Python and Machine Learning Community

   

⭐ If you found this project interesting, consider giving the repository a star on GitHub!
