🎓 Project Title: Mathematical Concepts of Linear Algebra and Optimization Using a Generative Artificial Intelligence Tool – Implementation of the Steepest Descent Method

This project was presented at the 1st Student and Mathematical Conference of the University of the Peloponnese, organized by the Department of Digital Systems on March 14, 2025. Its aim was to bridge theoretical mathematical concepts with modern artificial intelligence technologies through the practical implementation of the Steepest Descent algorithm.

🎯 Project Objective
The purpose of this work is to:

Highlight the connection between linear algebra, optimization, and artificial intelligence.

Demonstrate how the Steepest Descent method can be used to find the minimum of a function with two variables.

Showcase how a generative AI tool can support the educational process by automatically generating assessment material and enhancing conceptual understanding.

🧮 Code Overview
1️⃣ Steepest Descent Algorithm Implementation
The first code implements the Steepest Descent algorithm to minimize a two-variable function. The user is prompted to input:

Initial starting points 
(
𝑥
0
,
𝑦
0
)

Learning rate 
𝑎

Three termination criteria: 
𝑐
1
, 
𝑐
2
, and 
𝑐
3

The function to be minimized

The algorithm calculates partial derivatives and the gradient, updating the coordinates at each iteration. The process terminates when:

The gradient norm is smaller than 
𝑐
1

The distance between successive points is less than 
𝑐
2

The difference in function values between iterations is less than 
𝑐
3

Or when the maximum number of iterations (1000) is exceeded

The user receives:

A visual representation of the algorithm’s path in both 2D and 3D graphs

Final values of the variables and the function

A message indicating which termination criterion was satisfied

2️⃣ Automatic Question Generation Using Generative AI
The second code uses the OpenAI API to automatically generate multiple-choice questions based on the content of a PDF document. The process includes:

Extracting text from the PDF (e.g., theoretical background on AI or optimization)

Creating a prompt that asks the model to generate 10 questions in Greek

Assigning difficulty levels using a Likert scale (1–5)

Saving the generated questions to a text file

The user receives:

High-quality questions suitable for university-level learners

A range of difficulty levels for educational assessment

A demonstration of how AI can assist in content creation and learning

📊 What the Code Demonstrates to the User
The practical application of optimization theory

The dynamic path toward a function’s minimum

Interactive control over algorithm parameters and visualization

The integration of artificial intelligence in educational workflows

A clear connection between mathematical theory and modern AI tools, enhancing student engagement and understanding
