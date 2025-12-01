import re
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()

# credentials and keys. Add yours here 
CLIENT_ID = "CLIENT_ID"
CLIENT_SECRET = "CLIENT_SECRET"
USER_AGENT = "Project_Name:1.0 (by u/userName)"

# outputs 
OUT_DIR = Path(__file__).resolve().parent
CSV_OUT = OUT_DIR / "mental_health_influencer_metrics.csv"
JSON_OUT = OUT_DIR / "mental_health_influencer_data.json"

# configuration 
SUBREDDITS_TO_SCAN = [
# Add here, each subreddit in "" and separated by a ,
]

# Limit comments to save time
COMMENTS_PER_POST_LIMIT = 40 
POSTS_PER_SUB = 1000  

# 1. Medical/Disclaimer Metrics
DISCLAIMERS = [
# Add here, each disclaimer in "" and separated by a ,
]

# 2. Search Keywords 
# These are used to FIND the posts.
INFLUENCER_TERMS = [
# Add here, each term in "" and separated by a ,
]

RE_DISCLAIMER = re.compile(r"\b(" + "|".join(DISCLAIMERS) + r")\b", re.I)
# We create a wider regex for local analysis to catch more hits in the text
RE_INFLUENCER = re.compile(r"\b(" + "|".join(INFLUENCER_TERMS) + r")\b", re.I)

# helper functions 
def to_iso(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat()

def get_sentiment(text: str) -> float:
    """Returns the compound sentiment score (-1 to 1)."""
    if not text: return 0.0
    return vader.polarity_scores(text[:5000])['compound']

def check_content_flags(text: str) -> Dict:
    """Checks for disclaimers and influencer mentions."""
    return {
        "has_disclaimer": bool(RE_DISCLAIMER.search(text)),
        "mentions_social_media": bool(RE_INFLUENCER.search(text))
    }

def get_domain_type(url: str) -> str:
    """Categorize the link source."""
    if not url: return "text_only"
    if "reddit.com" in url: return "self_post"
    if any(x in url for x in ["ncbi.nlm.nih.gov", ".edu", "scholar", "science", "apa.org"]):
        return "academic/medical"
    if any(x in url for x in ["tiktok.com", "youtube.com", "instagram.com", "twitter.com", "x.com"]):
        return "social_media"
    return "other_web"

# core logic
def process_comments(submission) -> Tuple[List[Dict], float]:
    """Fetches comments and calculates average sentiment."""
    comment_data = []
    sentiments = []
    
    submission.comments.replace_more(limit=0)
    
    count = 0
    for comment in submission.comments.list():
        if count >= COMMENTS_PER_POST_LIMIT: break
        
        text = getattr(comment, "body", "")
        if text in ["[deleted]", "[removed]"]: continue
        
        c_sentiment = get_sentiment(text)
        sentiments.append(c_sentiment)
        
        comment_data.append({
            "id": comment.id,
            "body": text[:200], 
            "author": str(comment.author),
            "score": comment.score,
            "sentiment_score": c_sentiment
        })
        count += 1
        
    avg_sentiment = statistics.mean(sentiments) if sentiments else 0.0
    return comment_data, avg_sentiment

def analyze_post(s) -> Dict:
    """Extracts all metrics for a single post."""
    full_text = f"{s.title} {s.selftext}"
    
    sentiment_score = get_sentiment(full_text)
    flags = check_content_flags(full_text)
    engagement_score = s.score + s.num_comments
    
    comments_extracted, avg_comment_sentiment = process_comments(s)
    
    row = {
        "id": s.id,
        "created_utc": to_iso(s.created_utc),
        "subreddit": s.subreddit.display_name,
        "author": str(s.author),
        "title": s.title,
        "full_url": s.url,
        
        # Metrics
        "domain_category": get_domain_type(s.url),
        "post_length_chars": len(full_text),
        "has_disclaimer": flags['has_disclaimer'], 
        "mentions_influencers": flags['mentions_social_media'],
        "over_18": s.over_18,
        "upvote_score": s.score,
        "upvote_ratio": s.upvote_ratio,
        "num_comments": s.num_comments,
        "total_engagement": engagement_score,
        "post_sentiment": sentiment_score, 
        "avg_comment_sentiment": avg_comment_sentiment, 
        "sentiment_gap": sentiment_score - avg_comment_sentiment, 
        "top_comments_data": json.dumps(comments_extracted) 
    }
    return row

def main():
    if not CLIENT_ID:
        print("Please fill in CLIENT_ID and CLIENT_SECRET.")
        return

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        check_for_async=False
    )

    all_data = []
    
    print(f"[*] Starting scan. Metrics: Sentiment, Disclaimers, Engagement.")
    
    for sub_name in SUBREDDITS_TO_SCAN:
        try:
            print(f" -> Scanning r/{sub_name}...")
            sub = reddit.subreddit(sub_name)
            
            # Simplified query string creation to ensure hits
            query_string = " OR ".join(INFLUENCER_TERMS)
            
            # Fetch relevant posts
            posts = list(sub.search(query_string, limit=POSTS_PER_SUB))
            
            # If search returns nothing, try fetching 'hot' posts and filtering manually
            # This ensures we get data even if search is strict
            if len(posts) < 5:
                print(f"    (Search yielded few results, switching to 'Hot' scan...)")
                hot_posts = list(sub.hot(limit=100))
                # Add posts that match our regex locally
                posts.extend([p for p in hot_posts if RE_INFLUENCER.search(p.title + (p.selftext or ""))])
            
            unique_posts = {p.id: p for p in posts}.values()

            for post in unique_posts:
                data = analyze_post(post)
                all_data.append(data)
                print(f"    Processed: {post.title[:40]}... (Sent: {data['post_sentiment']:.2f})")
                
            time.sleep(2) 
            
        except Exception as e:
            print(f"Error in r/{sub_name}: {e}")

    # export files
    if not all_data:
        print("No data found.")
        return

    df = pd.DataFrame(all_data)
    
    csv_df = df.drop(columns=["top_comments_data"]) 
    csv_df.to_csv(CSV_OUT, index=False)
    
    with open(JSON_OUT, "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    print(f"\n[SUCCESS] Processed {len(df)} posts.")
    print(f"Metrics saved to: {CSV_OUT}")

if __name__ == "__main__":
    main()
reddit_final_mental_health.py
10 KB
