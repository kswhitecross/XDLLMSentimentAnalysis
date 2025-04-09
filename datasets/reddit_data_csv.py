import praw
import csv
import json

# Reddit API Credentials
reddit = praw.Reddit(
    client_id="Pr-QMMh8HpYZA482fARx7g",
    client_secret="C4nRcLG118R5gb0o4RonOwKCrMZKkA",
    user_agent="python:my_reddit_scraper:v1.0 (by u/ajc_617)",
    username="ajc_617",
    password="stewie_pigs0S"
)

#subreddits to get posts and comments for
subreddits = ["artificial", "politics", "conservative", "wallstreetbets", "gamingcirclejerk", "movies", "pcmasterrace", "vegan", "climatechange", "feminism", "mensrights"]
subreddits2 = ["funny", "askreddit", "gaming", "worldnews", "todayilearned", "awww", "music", "memes", "showerthoughts"]
merged_subreddits = subreddits + subreddits2

csv_rows = []


#used for debugging
i = 0
#loop through subreddits
for subreddit_name in merged_subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    #get top 10 posts for the last year for the subreddit
    top_10_posts = subreddit.top(time_filter="year", limit=10)
    for post in top_10_posts:
        # print(i)
        i += 1
        #first column is subreddit name, second is post title
        cur_csv_row = [subreddit_name, post.title]
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

file_path = '/reddit_posts_comments.csv'

# Open the file in write mode and create a CSV writer object
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    column_names = ["subreddit", "post_title", "comments"]
    writer.writerow(column_names)
    # Write all rows from the list of lists into the CSV file
    writer.writerows(csv_rows)

print(f"CSV file '{file_path}' has been created.")