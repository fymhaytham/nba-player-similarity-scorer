# NBA Player Similarity Scorer 🏀

A Python-based tool that calculates and visualizes statistical similarities between NBA players using machine learning techniques.

## Features

- **Data Retrieval**: Fetches real-time NBA player statistics using the `nba_api`.
- **Similarity Engine**: Utilizes Scikit-learn's `cosine_similarity` to find players with the most similar statistical profiles.
- **Normalization**: Scales statistics using `StandardScaler` for fair comparison across different categories.
- **Visualizations**: Generates comprehensive visual reports including:
  - Similarity ranking horizontal bar charts.
  - Statistical profile radar charts.
  - Grouped key stats comparisons (PTS, AST, REB).
  - Statistical heatmaps.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fymhaytham/nba-player-similarity-scorer.git
   cd nba-player-similarity-scorer
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nba_api
   ```

## Usage

Run the main script:

```bash
python main.py
```

Follow the prompt to enter a player name and season (e.g., `Stephen Curry 2024-25`).

## How it Works

The system calculates similarity based on the following key metrics:
- Points (PTS)
- Rebounds (REB)
- Assists (AST)
- Field Goal Percentage (FG_PCT)
- 3-Point Percentage (FG3_PCT)
- Free Throw Percentage (FT_PCT)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TOV)

It uses Cosine Similarity on normalized data to determine how "close" one player's statistical output is to another.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nba_api
