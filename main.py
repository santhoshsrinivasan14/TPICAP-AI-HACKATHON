from vector import Retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import csv
import datetime
import pandas as pd
import os

# 1. Feedback file path
FEEDBACK_FILE = "user_feedback.csv"

# 2. Load user's past liked courses
def load_user_likes(user_id):
    if not os.path.exists(FEEDBACK_FILE):
        return []
    df = pd.read_csv(FEEDBACK_FILE)
    likes = df[(df['user_id'] == user_id) & (df['selected_title'].notna())]['selected_title'].tolist()
    return likes

# 3. Save feedback to file
def save_feedback(user_id, query, reviews, selected_title):
    write_header = not os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["user_id", "query", "reviews", "selected_title", "timestamp"])
        writer.writerow([
            user_id, query, reviews.replace('\n', ' | '), selected_title, str(datetime.datetime.now())
        ])

# 4. Main RAG setup
retriever = Retriever()
model = OllamaLLM(model="phi3")  # or use "llama2:7b" if phi3 doesn't work for you!
template = """
You are an expert course advisor.
Here are some relevant courses: {reviews}
Here is the user's question: {question}
Recommend 3 courses with a one-sentence reason each.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("\n🧠 AI Course Recommender with Memory (type 'q' to quit)\n")
user_id = input("Enter your user ID: ")

user_likes = load_user_likes(user_id)
if user_likes:
    print(f"👀 Welcome back, {user_id}! Last time, you liked: {user_likes}")
else:
    print("👋 Welcome! Let's find you some courses.")

while True:
    print("\n-------------------------------")
    question = input("Ask your question: ")
    if question.strip().lower() == "q":
        break

    reviews_raw = retriever.invoke(question)
    # Split the reviews into a list of (title, desc) pairs
    review_lines = [line for line in reviews_raw.split("\n\n") if line.strip()]
    parsed_reviews = []
    for r in review_lines:
        if r.startswith("**"):
            # Format: **Course Title**: Description
            try:
                title = r.split("**")[1]
                desc = r.split("**: ")[-1]
            except Exception:
                title = r
                desc = ""
            parsed_reviews.append((title, desc))
        else:
            parsed_reviews.append((r, ""))

    # BOOST: Move liked courses to the top
    boosted = []
    normal = []
    for title, desc in parsed_reviews:
        if title in user_likes and title.strip():
            boosted.append((title, desc))
        else:
            normal.append((title, desc))
    all_reviews = boosted + normal

    # For the LLM, just use the concatenated string
    reviews_boosted = "\n\n".join([f"**{t}**: {d}" for t, d in all_reviews])
    result = chain.invoke({"reviews": reviews_boosted, "question": question})

    print("\n📚 Recommendations:\n")
    # Print boosted results with a tag
    for idx, (title, desc) in enumerate(all_reviews, 1):
        like_tag = " (You liked this before!)" if title in user_likes else ""
        print(f"{idx}. {title}{like_tag}")
        if desc:
            print(f"   {desc}")
        print()

    selected_title = input("\nWhich course above do you like? (paste course title, or leave blank to skip): ")
    if selected_title and selected_title not in user_likes:
        user_likes.append(selected_title)
    save_feedback(user_id, question, reviews_boosted, selected_title if selected_title else "")


print("👋 Goodbye!")
