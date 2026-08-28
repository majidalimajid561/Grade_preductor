🎓 Student Grade Predictor

A machine learning web app that predicts a student's final exam grade based on study habits and past performance.

Built with Python, Scikit-learn, and Streamlit.

🔗 Live Demo

[👉 Click here to try the app](https://gradepreductor-vtwmchssraqttes4pgomim.streamlit.app/)


📌 What This App Does

You enter three inputs:

Input	Description
Daily Study Hours	How many hours you study per day
Attendance %	Your class attendance percentage
Past Exam Score	Your score in the previous exam

The app instantly predicts your final grade out of 100 and tells you:

Whether you are at risk, passing, or excelling
Which factor affects your grade the most
How much one extra study hour would improve your grade
🧠 Machine Learning Details
Detail	Value
Algorithm	Linear Regression
Library	Scikit-learn
Features	Study Hours, Attendance, Past Score
Target	Final Grade (0–100)
Train/Test Split	80% / 20%
Evaluation Metric	MAE, R² Score
How it works

The model learns the relationship between study habits and grades from training data. It finds the best linear equation:

Grade = (study_hours × w1) + (attendance × w2) + (past_score × w3) + bias

The weights (w1, w2, w3) are learned automatically during training — no manual rules written.

🛠️ Tech Stack

Python 3.10
Streamlit — web app framework
Scikit-learn — machine learning model
NumPy — data generation and array operations
Pandas — data handling

🚀 Run Locally

1. Clone the repository
bash
git clone https://github.com/majidalimajid561/Grade_preductor.git
cd grade-predictor
2. Create and activate conda environment
bash
conda create -n mlenv python=3.10
conda activate mlenv
3. Install dependencies
bash
pip install streamlit scikit-learn numpy pandas
4. Run the app
bash
streamlit run app.py


App will open at http://localhost:8501


📁 Project Structure

grade-predictor/
│
├── app.py          ← main Streamlit app
├── README.md       ← this file


📊 Model Performance
Metric	Value
Mean Absolute Error (MAE)	~2.11 grade points
R² Score	~0.96


💡 What I Learned Building This
How to implement Linear Regression using Scikit-learn
How to evaluate a model using MAE and R² score
How to build and deploy a web app using Streamlit
How train/test split prevents overfitting
How model coefficients reveal which features matter most

👤 Author

Majid Mehmood

[GitHub](https://github.com/majidalimajid561/)

[LinkedIn](https://www.linkedin.com/in/majid-mehmood-4286533ba/)

📄 License

This project is open source and available under the MIT License.
