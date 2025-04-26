import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="🍉 Fruit Classifier", layout="centered")

# Load model


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fruit_model.h5')


model = load_model()

# Fruit label map
fruitMap = {
    0: 'Apple Braeburn', 1: 'Apple Granny Smith', 2: 'Apricot', 3: 'Avocado',
    4: 'Banana', 5: 'Blueberry', 6: 'Cactus fruit', 7: 'Cantaloupe', 8: 'Cherry',
    9: 'Clementine', 10: 'Corn', 11: 'Cucumber Ripe', 12: 'Grape Blue', 13: 'Kiwi',
    14: 'Lemon', 15: 'Limes', 16: 'Mango', 17: 'Onion White', 18: 'Orange',
    19: 'Papaya', 20: 'Passion Fruit', 21: 'Peach', 22: 'Pear', 23: 'Pepper Green',
    24: 'Pepper Red', 25: 'Pineapple', 26: 'Plum', 27: 'Pomegranate', 28: 'Potato Red',
    29: 'Raspberry', 30: 'Strawberry', 31: 'Tomato', 32: 'Watermelon'
}

# Sample nutrition data, facts, and recipes for all fruits
fruit_details = {
    'Apple Braeburn': {'Nutrition': {'Calories': 52, 'Vitamin C (%)': 8}, 'Fact': 'Braeburn apples are crisp and have a strong flavor.', 'Recipe': 'Apple Pie or Apple Salad.'},
    'Apple Granny Smith': {'Nutrition': {'Calories': 57, 'Vitamin C (%)': 12}, 'Fact': 'Granny Smith apples are tart and firm.', 'Recipe': 'Apple Crisp or Green Apple Smoothie.'},
    'Apricot': {'Nutrition': {'Calories': 48, 'Vitamin C (%)': 10}, 'Fact': 'Apricots are rich in antioxidants and Vitamin A.', 'Recipe': 'Apricot Jam or Apricot Salad.'},
    'Avocado': {'Nutrition': {'Calories': 160, 'Vitamin C (%)': 17}, 'Fact': 'Avocados are rich in healthy fats and fiber.', 'Recipe': 'Guacamole or Avocado Toast.'},
    'Banana': {'Nutrition': {'Calories': 89, 'Vitamin C (%)': 15}, 'Fact': 'Bananas are rich in potassium.', 'Recipe': 'Banana Bread or Smoothie.'},
    'Blueberry': {'Nutrition': {'Calories': 57, 'Vitamin C (%)': 16}, 'Fact': 'Blueberries are packed with antioxidants.', 'Recipe': 'Blueberry Muffins or Smoothie.'},
    'Cactus fruit': {'Nutrition': {'Calories': 50, 'Vitamin C (%)': 20}, 'Fact': 'Cactus fruits are known for their refreshing flavor.', 'Recipe': 'Cactus Fruit Salad or Jams.'},
    'Cantaloupe': {'Nutrition': {'Calories': 34, 'Vitamin C (%)': 58}, 'Fact': 'Cantaloupes are sweet and juicy melons.', 'Recipe': 'Cantaloupe Salad or Smoothie.'},
    'Cherry': {'Nutrition': {'Calories': 50, 'Vitamin C (%)': 16}, 'Fact': 'Cherries are rich in antioxidants and anti-inflammatory properties.', 'Recipe': 'Cherry Pie or Cherry Jam.'},
    'Clementine': {'Nutrition': {'Calories': 47, 'Vitamin C (%)': 100}, 'Fact': 'Clementines are small and easy-to-peel citrus fruits.', 'Recipe': 'Clementine Sorbet or Juice.'},
    'Corn': {'Nutrition': {'Calories': 96, 'Vitamin C (%)': 10}, 'Fact': 'Corn is a versatile crop used in various dishes worldwide.', 'Recipe': 'Corn Salad or Grilled Corn.'},
    'Cucumber Ripe': {'Nutrition': {'Calories': 16, 'Vitamin C (%)': 3}, 'Fact': 'Cucumbers are low in calories and hydrating.', 'Recipe': 'Cucumber Salad or Pickled Cucumbers.'},
    'Grape Blue': {'Nutrition': {'Calories': 69, 'Vitamin C (%)': 18}, 'Fact': 'Blue grapes are a great source of fiber and antioxidants.', 'Recipe': 'Grape Jam or Smoothie.'},
    'Kiwi': {'Nutrition': {'Calories': 41, 'Vitamin C (%)': 93}, 'Fact': 'Kiwis are packed with Vitamin C and fiber.', 'Recipe': 'Kiwi Sorbet or Smoothie.'},
    'Lemon': {'Nutrition': {'Calories': 29, 'Vitamin C (%)': 64}, 'Fact': 'Lemons are known for their tangy flavor and high vitamin C content.', 'Recipe': 'Lemonade or Lemon Meringue Pie.'},
    'Limes': {'Nutrition': {'Calories': 30, 'Vitamin C (%)': 35}, 'Fact': 'Limes are commonly used to enhance flavor in many dishes.', 'Recipe': 'Lime Pie or Lime Marinade.'},
    'Mango': {'Nutrition': {'Calories': 60, 'Vitamin C (%)': 60}, 'Fact': 'Mangoes are known as the king of fruits.', 'Recipe': 'Mango Lassi or Mango Salsa.'},
    'Onion White': {'Nutrition': {'Calories': 40, 'Vitamin C (%)': 8}, 'Fact': 'Onions add flavor to a variety of dishes.', 'Recipe': 'Onion Rings or Onion Soup.'},
    'Orange': {'Nutrition': {'Calories': 47, 'Vitamin C (%)': 88}, 'Fact': 'Oranges are high in Vitamin C.', 'Recipe': 'Orange Juice or Fruit Salad.'},
    'Papaya': {'Nutrition': {'Calories': 59, 'Vitamin C (%)': 144}, 'Fact': 'Papayas are tropical fruits known for their sweet flavor.', 'Recipe': 'Papaya Smoothie or Papaya Salad.'},
    'Passion Fruit': {'Nutrition': {'Calories': 97, 'Vitamin C (%)': 30}, 'Fact': 'Passion fruits are aromatic and tangy in flavor.', 'Recipe': 'Passion Fruit Juice or Sorbet.'},
    'Peach': {'Nutrition': {'Calories': 39, 'Vitamin C (%)': 13}, 'Fact': 'Peaches are juicy and sweet fruits.', 'Recipe': 'Peach Cobbler or Peach Ice Cream.'},
    'Pear': {'Nutrition': {'Calories': 57, 'Vitamin C (%)': 8}, 'Fact': 'Pears are known for their soft texture.', 'Recipe': 'Pear Salad or Pear Jam.'},
    'Pepper Green': {'Nutrition': {'Calories': 20, 'Vitamin C (%)': 150}, 'Fact': 'Green peppers are a rich source of Vitamin C.', 'Recipe': 'Stuffed Peppers or Grilled Peppers.'},
    'Pepper Red': {'Nutrition': {'Calories': 31, 'Vitamin C (%)': 157}, 'Fact': 'Red peppers are loaded with antioxidants.', 'Recipe': 'Roasted Peppers or Pepper Salsa.'},
    'Pineapple': {'Nutrition': {'Calories': 50, 'Vitamin C (%)': 80}, 'Fact': 'Pineapples are a tropical fruit with a tangy-sweet flavor.', 'Recipe': 'Pineapple Smoothie or Grilled Pineapple.'},
    'Plum': {'Nutrition': {'Calories': 46, 'Vitamin C (%)': 10}, 'Fact': 'Plums are sweet and juicy fruits.', 'Recipe': 'Plum Jam or Plum Pie.'},
    'Pomegranate': {'Nutrition': {'Calories': 83, 'Vitamin C (%)': 17}, 'Fact': 'Pomegranates are packed with antioxidants.', 'Recipe': 'Pomegranate Salad or Juice.'},
    'Potato Red': {'Nutrition': {'Calories': 70, 'Vitamin C (%)': 28}, 'Fact': 'Red potatoes are starchy and perfect for roasting.', 'Recipe': 'Roasted Potatoes or Potato Salad.'},
    'Raspberry': {'Nutrition': {'Calories': 52, 'Vitamin C (%)': 54}, 'Fact': 'Raspberries are full of fiber and antioxidants.', 'Recipe': 'Raspberry Jam or Smoothie.'},
    'Strawberry': {'Nutrition': {'Calories': 33, 'Vitamin C (%)': 97}, 'Fact': 'Strawberries are a great source of antioxidants.', 'Recipe': 'Strawberry Jam or Strawberry Pie.'},
    'Tomato': {'Nutrition': {'Calories': 18, 'Vitamin C (%)': 40}, 'Fact': 'Tomatoes are rich in lycopene, an antioxidant.', 'Recipe': 'Tomato Soup or Tomato Salad.'},
    'Watermelon': {'Nutrition': {'Calories': 30, 'Vitamin C (%)': 17}, 'Fact': 'Watermelon is mostly water and very hydrating.', 'Recipe': 'Watermelon Salad or Smoothie.'}
}

# Team members
team_members = ["Alice Johnson", "Bob Smith", "Charlie Davis", "David Lee"]

st.title("🍓 Fruit Classifier")

# Project details
st.markdown("""
### Project Overview
This Fruit Classifier app uses machine learning models to classify fruits based on uploaded images. It provides information about the fruit, including fun facts, nutrition details, and recipe suggestions for each fruit. 

### Team Members
- Alice Johnson
- Bob Smith
- Charlie Davis
- David Lee
""")

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[-5:][::-1]
    top_labels = [fruitMap[i] for i in top_indices]
    top_scores = [prediction[i] * 100 for i in top_indices]

    # Save to prediction history
    st.session_state.history.append(f"{top_labels[0]} ({top_scores[0]:.2f}%)")

    # Main Prediction
    st.subheader("Prediction:")
    st.success(f"{top_labels[0]} ({top_scores[0]:.2f}% confidence)")

    # Warn if model is unsure
    if top_scores[0] < 60:
        st.warning(
            "⚠️ Model confidence is low. Try another image or different lighting.")

    # Nutrition Info
    if top_labels[0] in fruit_details and 'Nutrition' in fruit_details[top_labels[0]]:
        st.subheader("🍽️ Nutritional Info")
        st.json(fruit_details[top_labels[0]]['Nutrition'])

    # Recipe
    if top_labels[0] in fruit_details and 'Recipe' in fruit_details[top_labels[0]]:
        st.subheader("🍳 Recipe Suggestion")
        st.write(fruit_details[top_labels[0]]['Recipe'])

    # Fun Facts
    if top_labels[0] in fruit_details and 'Fact' in fruit_details[top_labels[0]]:
        st.subheader("🧠 Fun Fact")
        st.write(fruit_details[top_labels[0]]['Fact'])

    # Bar chart of top 5 predictions
    fig, ax = plt.subplots()
    ax.barh(top_labels, top_scores, color='skyblue')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top 5 Predicted Fruits')
    st.pyplot(fig)

# History Panel
st.sidebar.title("🕓 Prediction History")
for item in reversed(st.session_state.history[-5:]):
    st.sidebar.write(item)
