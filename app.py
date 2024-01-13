import streamlit as st
from deepface import DeepFace
import cv2
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
model = DeepFace.build_model("Emotion")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dark_theme = """
    <style>
        body {
            color: white;
            background-color: #1E1E1E; /* Dark background color */
        }
        .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
            color: black; /* Text color for input elements */
            background-color: #D3D3D3; /* Background color for input elements */
        }
        /* Add more specific styling as needed for different Streamlit elements */
    </style>
"""

def map_genres_to_emotions(genres, emotion_genre_mapping):
        emotions = set()
        for genre in genres.split(', '):
            for emotion, mapped_genres in emotion_genre_mapping.items():
                if genre in mapped_genres:
                    emotions.add(emotion)
        return ', '.join(emotions)

def select_best5(recommendations):
    max_val = recommendations.max()
    movie_list = []

    for movie in recommendations.index:
        if (recommendations[movie] == max_val).any():
            movie_list.append(movie)
    return movie_list
    

def recommended_movies(emotion):

    # Replace 'your_file_path.csv' with the actual path to your CSV file
    file_path = 'mymoviedb.csv'

    # Read the CSV file into a DataFrame
    movies_df = pd.read_csv(file_path,lineterminator='\n')

    emotion_genre_mapping = {
    'Happy': ['Animation', 'Family', 'Comedy', 'Romance'],
    'Sad': ['Drama', 'Romance', 'War'],
    'Angry': ['Action', 'Thriller', 'Crime'],
    'Surprised': ['Mystery', 'Science Fiction', 'Fantasy'],
    'Fearful': ['Horror', 'Thriller', 'Mystery'],
    'Disgusted': ['Horror', 'Thriller', 'Crime']
    }
    # Add Emotion column to the dataset
    movies_df['Emotion'] = movies_df['Genre'].apply(lambda x: map_genres_to_emotions(x, emotion_genre_mapping))
    movies_df['Features'] = movies_df['Genre'] + ', ' + movies_df['Emotion']

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Features'])

    # Calculate cosine similarity between movies
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    def recommend_movies_based_on_emotion(user_emotion, movies_df, cosine_similarity):
        # Transform user emotion into a TF-IDF vector
        user_vector = tfidf_vectorizer.transform([user_emotion])
        cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Calculate cosine similarity between user emotion and movies
        similarity_scores = linear_kernel(user_vector, tfidf_matrix).flatten()

        # Rank movies based on similarity scores
        rankings = pd.Series(similarity_scores, index=movies_df['Title']).sort_values(ascending=False)

        return rankings

    recommendations = recommend_movies_based_on_emotion(emotion, movies_df, cosine_similarity)
    top5 = select_best5(recommendations)
    return top5




# Display the HTML code to set the dark theme
st.markdown(dark_theme, unsafe_allow_html=True)
selected = st.button("Open Camera")
if selected :
    text = st.text("Detecting Emotion...")
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while time.time() - start_time < 6:
        # Capture the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show the webcam feed
        cv2.imshow('Adjust Your Face', frame)

        # Small delay to allow for key press detection
        cv2.waitKey(1)
    text.empty()
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = ""
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]
        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

         # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    st.image(frame)
    st.text(f"Detected Emotion: {emotion}")
    
    top5 = recommended_movies(emotion)
    random.shuffle(top5)
    st.title("Here are few recommended movies for you!")
    for i in range(5):
        st.text(top5[i])