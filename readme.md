![Fruit Classifier Banner](https://images.unsplash.com/photo-1574226516831-e1dff420e8f8?auto=format&fit=crop&w=1350&q=80)

# 🍉 Fruit Classifier App

Welcome to the **Fruit Classifier**!  
This is a fun and interactive machine learning project built using **TensorFlow** and **Streamlit**.  
Upload any fruit image, and the app will predict the type of fruit along with **fun facts**, **nutrition info**, and **recipe suggestions**! 🍎🍌🍇

---

## 📸 Features

- Upload an image and classify it among **33 different fruits**.
- View **Top 5 predictions** with confidence scores.
- Get **Nutrition information** (calories, Vitamin C %, etc.).
- Read **Fun facts** about the predicted fruit.
- Explore **Easy recipe ideas** for the fruit.
- See a **bar chart** of prediction probabilities.
- **Prediction history** saved for quick reference.
- **Team Members** and Project details included.

---

## 📚 Technologies Used

- [Streamlit](https://streamlit.io/) – Frontend UI
- [TensorFlow / Keras](https://www.tensorflow.org/) – Machine Learning model
- [Pillow (PIL)](https://python-pillow.org/) – Image processing
- [Matplotlib](https://matplotlib.org/) – Chart plotting
- [NumPy](https://numpy.org/) – Data manipulation

---

## 🏧 Project Structure

```plaintext
fruit-classifier/
|
|├── models/
|   └── fruit_model.h5
|
|├── images/
|   └── sample_fruits/
|
|├── app.py
|
|├── requirements.txt
|
└── README.md
```

---

## 🚀 How to Run the Project

1. **Clone the repository** or **Download** the project files.

2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Upload a fruit image** and see the magic! ✨


---

## 📍 Sample Fruits Supported

> Some examples:  
> Apple, Banana, Mango, Blueberry, Avocado, Watermelon, Papaya, Kiwi, Orange, Strawberry, Raspberry, Tomato, and many more!

(Fully supports **33 fruits** 🍉🍒🍍🥝🍑)

---

## 📌 Notes

- For best results, use **clear images** with **good lighting**.
- Some **random Google images** might not work perfectly if they are blurry, edited, or very different from training data.
- You can improve accuracy by training the model with **more diverse images** in the future!

---

# 🏁 Enjoy Classifying Fruits! 🍓🍋🥭🍇

---

> _Built with ❤️ by an awesome team!_
