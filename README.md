# Weekly Reddit Trends

Personal, non-commercial tool for creating two private weekly Reddit digests:

- **AI & Tech Weekly**
- **Personal Weekly**

The app is intended to be read-only. It will collect a limited set of recent public posts and selected top-level comments from a fixed allowlist of subreddits, rank and cluster them, and use an existing LLM API for classification and summarization.

## Planned flow

Reddit API → ranking / shortlist → topic clustering → LLM analysis → static HTML reports

The reports will be hosted privately and will link back to representative Reddit threads.

## Data use

- Read-only Reddit access
- No posting, commenting, voting, messaging or moderation
- No user profiling
- No resale or bulk redistribution of Reddit data
- No AI/ML model training or fine-tuning
- Raw Reddit content retained only as long as needed for processing

## Project status

Early development / API access application stage.

## Configuration

- `config/communities.yaml` – monitored subreddits, profiles, categories and tiers
- `config/ranking.yaml` – v0 ranking, radar and report-selection rules

## Security

API keys, passwords and deployment credentials are not stored in this repository. They will be provided through environment variables / GitHub Secrets.
