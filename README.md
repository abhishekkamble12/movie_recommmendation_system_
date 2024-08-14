# movie_recommmendation_system_
https://github.com/abhishekkamble12/movie_recommmendation_system_
 ![Screenshot 2024-08-14 213210](https://github.com/user-attachments/assets/1fd58d35-88be-4c26-8c75-16860c8fdb66)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv')

# Select the relevant columns (you can choose others if needed)
df = df[['Movie_Title', 'Movie_Genre']]

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Check the data
print(f"Train set: {train.shape}")
print(f"Test set: {test.shape}")

# Vectorize the 'Movie_Genre' column using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf.fit_transform(train['Movie_Genre'])

# Calculate the cosine similarity matrix for the training set
cosine_sim_train = cosine_similarity(tfidf_matrix_train, tfidf_matrix_train)

def get_recommendations_by_genre(genre, n_recommendations=5):
    # Vectorize the input genre
    genre_tfidf = tfidf.transform([genre])

    # Calculate the cosine similarity with all genres in the training set
    cosine_sim_genre = cosine_similarity(genre_tfidf, tfidf_matrix_train)

    # Get the top n recommendations
    sim_scores = list(enumerate(cosine_sim_genre[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:n_recommendations]

    # Get the movie indices and titles
    movie_indices = [i[0] for i in sim_scores]
    return train['Movie_Title'].iloc[movie_indices], sim_scores

# Input from the user
user_genre = input("Enter a genre: ")

# Get recommendations
recommended_movies, scores = get_recommendations_by_genre(user_genre, n_recommendations=5)

# Output the recommendations
print("Recommended Movies:")
for movie, score in zip(recommended_movies, scores):
    print(f"{movie} (Similarity Score: {score[1]:.2f})")

# Plot the similarity scores of the recommended movies
def plot_recommendations(recommended_movies, scores):
    movie_titles = recommended_movies.tolist()
    similarity_scores = [score[1] for score in scores]

    sns.barplot(x=similarity_scores, y=movie_titles, palette='viridis')
    plt.title('Recommended Movies Based on Genre Similarity')
    plt.xlabel('Similarity Score')
    plt.ylabel('Movie Title')
    plt.show()

# Visualize the recommendations
plot_recommendations(recommended_movies, scores)
