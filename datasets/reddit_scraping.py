import praw
import csv
import json
import os 
# Reddit API Credentials
client_id = os.environ.get("PRAW_CLIENT_ID", default=None)
client_secret = os.environ.get("PRAW_CLIENT_SECRET", default=None)
username = os.environ.get("PRAW_REDDIT_USERNAME", default=None)
pw = os.environ.get("PRAW_REDDIT_PW", default=None)

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="script: research scraper (by 685 spring 25 team)",
    username=username,
    password=pw
)

#subreddits to get posts and comments for
subreddits = ["funny", "askreddit", "gaming", "worldnews", "todayilearned", "awww", "music", "memes", "movies", "showerthoughts"]
csv_rows = []


#used for debugging
i = 0
#loop through subreddits
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    #get top 10 posts for the last year for the subreddit
    top_100_posts = subreddit.top(time_filter="year", limit=100)
    for post in top_100_posts:
        print(i)
        i += 1
        #first column is subreddit name, second is post title
        cur_csv_row = [subreddit_name, post.title, post.selftext.replace("\n", " ")]
        #used to load in all comments
        post.comments.replace_more(limit=0)
        # Get the list of all comments
        comments = post.comments.list()
        # Sort comments by score (upvotes)
        sorted_comments = sorted(comments, key=lambda x: x.score, reverse=True)
        # Get the top 100 comments (limit to 100 if there are fewer comments)
        top_100_comments = sorted_comments[:100]
        #loop through comments and append bodies to cur_csv_row
        cur_comments = []
        for comment in top_100_comments:
            #skipping deleted comments
            if comment.body == "[deleted]":
                continue
            #need to replace new lines in the comments with spaces
            cur_comments.append(comment.body.replace("\n", " "))
        #now append the row for the post, ensure_ascii=False converts unicode to proper values
        cur_csv_row.append(json.dumps(cur_comments, ensure_ascii=False))
        csv_rows.append(cur_csv_row)

file_path = './reddit_top_100_posts_scraped.csv'

# Open the file in write mode and create a CSV writer object
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    column_names = ["subreddit", "post_title", "post_body", "comments"]
    writer.writerow(column_names)
    # Write all rows from the list of lists into the CSV file
    writer.writerows(csv_rows)

print(f"CSV file '{file_path}' has been created.")