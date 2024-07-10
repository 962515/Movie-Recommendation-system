import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import messagebox


movie = pd.read_csv("D:/internship/codsoft/recomm/tmdb_5000_movies.csv")
rating = pd.read_csv("D:/internship/codsoft/recomm/tmdb_5000_credits.csv")


movie['overview'] = movie['overview'].fillna("")


tfidf = TfidfVectorizer(stop_words="english")


tfidf_matrix = tfidf.fit_transform(movie['overview'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


index = pd.Series(movie.index, index=movie['original_title']).drop_duplicates()


def recomm():
    title = title_entry.get().title() 
    if title not in index:
        messagebox.showerror("Error", f"'{title}' not found in the movie database.")
        return
    
    idx = index[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:11]  # Get the scores of the 10 most similar movies
    
    sim_index = [i[0] for i in sim_score]
    recommended_movies = movie["original_title"].iloc[sim_index]
    result_text.set("\n".join(recommended_movies))


root = tk.Tk()
root.title("Movie Recommender")

tk.Label(root, text="Enter movie title:").pack(pady=5)

title_entry = tk.Entry(root, width=50)
title_entry.pack(pady=5)

recommend_button = tk.Button(root, text="Recommend", command=recomm)
recommend_button.pack(pady=5)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify="left")
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
