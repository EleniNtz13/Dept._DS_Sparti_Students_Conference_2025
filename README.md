## 🎓 Optimization in Machine Learning: Analysis and Implementation of the Steepest Descent Algorithm and Use of API for Creating Educational Questions

This project was developed as part of my undergraduate thesis at University of Peloponnese. This project was presented at the **1st Student and Mathematical Conference** of the UoP 🏫, organized by the Department of Digital Systems 💻 in **Sparta** on *March 14, 2025*. Its aim was to bridge theoretical mathematical concepts with modern artificial intelligence technologies through the practical implementation of the Steepest Descent algorithm.

### 🎯 Project Objective
The purpose of this work is to:

- Highlight the connection between *linear algebra 📚*, *optimization 📈*, and *artificial intelligence 🤖*.

- Demonstrate how the **Steepest Descent method** can be used to find the minimum of a function with two variables.

- Showcase how a **Generative AI Tool** can support the educational process by automatically generating assessment material and enhancing conceptual understanding.

---

### 🧐 Code Overview 

#### 1️⃣ Steepest Descent Algorithm Implementation (Steepest_Descent.py)

The first code implements the Steepest Descent algorithm to minimize a two-variable function. 

👨🏻‍💻 The user is prompted to input:

- Initial starting points: $(x_0, y_0)$,

- Learning rate: $a$, a number that controls the size of each step an optimization algorithm takes when updating its parameter. It tells how big of a correction to make in the direction indicated by the gradient:

   - If it’s too *large* → the algorithm may overshoot the minimum and diverge.

   - If it’s too *small* → the steps become tiny, and convergence becomes very slow.

- Three constant termination criteria: $c_1, c_2, c_3$,


🧮 The algorithm calculates partial derivatives and the gradient, updating the coordinates at each iteration. The process terminates when:

- The gradient norm is smaller than $𝑐_1$:
  $\nabla f(x_{k+1}) < c_1\$

- The distance between successive points is less than $𝑐_2$:
  $\ x_{k+1} - x_k < c_2\$

- The difference in function values between iterations is less than $𝑐_3$:
  $f(x_{k+1}) - f(x_k) < c_3$

- Or when the maximum number of iterations (1000) is exceeded


🙍🏻‍♂️ The user receives:

- A visual representation of the algorithm’s path in both 2D and 3D graphs

- Final values of the variables and the function

- A message indicating which termination criterion was satisfied



#### 2️⃣ Question Generation Based on Algorithm Parameters (API_Conf_DS_2025.py)
The second code uses the OpenAI API to generate multiple-choice questions *based on the algorithm’s logic and structure 🧠*, without external documents. The questions are created in Greek and are tailored to the Steepest Descent method, using a predefined prompt that includes theoretical context and algorithmic details. Each question is assigned a difficulty level from 1 to 5, and the output is saved in a text file.

#### 3️⃣ Question Generation Based on PDF Input (API_PDF.py)
The third code also uses the OpenAI API, but with a different approach: it enhances the language model’s input *by providing a PDF document 📝* containing theoretical or practical content related to the Steepest Descent method. The model extracts relevant information from the PDF and generates questions accordingly, allowing for deeper contextualization and more accurate alignment with the source material.

---

### 📊 What the Code Demonstrates to the User
- The practical application of optimization theory

- The dynamic path toward a function’s minimum

- Interactive control over algorithm parameters and visualization

- The integration of artificial intelligence in educational workflows

- A clear connection between mathematical theory and modern AI tools, enhancing student engagement and understanding


### 🧠 Tools Used for assistance and code optimization

- ChatGPT AI Tool  
- DeepSeek
- Google Colab 
  
