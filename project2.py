# # Important Libraryies
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import missingno as msno
# import plotly.express as px
# from sklearn.preprocessing import LabelEncoder

# from PIL import Image

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from PIL import Image


# # Add Image
# st.write('''
# ## **Add Image**''')

# image1 =Image.open("Kidny diseas streamlit App using ML models (3).png")
# st.image(image1, caption='Kidny diseas streamlit App using ML models', use_column_width=True)

# # WEBAPP TITLE
# st.title("Kidney Disease Prediction")

# # Data set k name ak button ma rakh kar sidebaar pay lagana
# dataset_name = st.sidebar.title("KIDNEY DISEASE")

# # Or isi k nichy classifier k nam ak dabay ma dal do
# classifier_name = st.sidebar.selectbox("Select Classifier", ("LogisticRegression", "Random Forest","GradientBoosting","DecisionTree","KNeighborsClassifier"))
# #For Import Dataset
# df = pd.read_csv('kidney.csv')


# st.write(df.head(10))

# st.write('''
# ## **COLUMNS DETAIL**''')

# ('''
# |Full Names | Abbrivations |Full Names | Abbrivations |
# |--------|----------|--------|----------|
# |age | age | white blood cell count | wc | 
# | red blood cell count |  rc |pus cell clumps | pcc |
# | blood glucose random | bgr |  pus cell | pc |
# | blood urea | bu | red blood cells | rbc | 
# | serum creatinine  | sc |sugar  | su |
# | sodium | sod | albumin | al |
# | potassium | pot |specific gravity | sg | 
# | hemoglobin | hemo |blood pressure | bp |
# | packed cell volume | pcv | hypertension | htn |
# | diabetes mellitus | dm | coronary artery disease | cad |
# | appetite |  appet | pedal edema |  pe |
# | anemia | ane | classification | classification |''')

# # Headings
# st.sidebar.header("Patient Data")
# st.subheader("Descriptive Statistics Of Data")
# st.write(df.describe())


# st.write('''
# ## **NULL VALUES**''')
# # Watch Null Values

# st.write(df.isnull().sum())



# st.write('''
# ## **PLOTS**''')

# # PLOTS
# plt.figure(figsize=(10,8)) 
# fig1 = sns.heatmap(df.corr()) 
# plt.show()
# st.write(fig1)

# #Heatmap
# fig2 = px.imshow(df,text_auto=True,aspect='auto')
# fig2.show()
# st.write(fig2)

# #Line Plot
# fig3 = plt.figure(figsize=(10,8))
# sns.lineplot(data=df,y='pc',x='bp')
# st.pyplot(fig3)


# # Each bar in the following bar plot is representing a column within our dataset. Height of the bar indicates how many non-null values are present in that particular column.

# fm1 = msno.bar(df)
# st.pyplot(fm1.figure)

# # This matrix plot provides a colour fill for each column of dataset. The plot is shaded where data is present. And white display means null values.

# fm2 = msno.matrix(df)
# st.pyplot(fm2.figure)

# # To check the correlation between of the null values between each column, heatmap is plotted.

# fm3 = msno.heatmap(df)
# st.pyplot(fm3.figure)


# # Drop Some Columns
# st.write(df.drop(columns="rbc",inplace=True))
# st.write(df.drop(columns="pc",inplace=True))
# st.write(df.drop(columns="pcc",inplace=True))
# st.write(df.drop(columns="ba",inplace=True))
# st.write(df.drop(columns="htn",inplace=True))
# st.write(df.drop(columns="dm",inplace=True))
# st.write(df.drop(columns="cad",inplace=True))
# st.write(df.drop(columns="appet",inplace=True))
# st.write(df.drop(columns="pe",inplace=True))
# st.write(df.drop(columns="ane",inplace=True))


# st.write('''
# ## ** SEE THE DROP COLUMNS**''')

# st.write(df.head(10))

# # deeling with missing values
# df['age']=df.age.fillna(value=df['age'].mean())
# df['bp']=df.bp.fillna(value=df['bp'].mean())
# df['al']=df.al.fillna(value=df['al'].mean())
# df['su']=df.su.fillna(value=df['su'].mean())
# df['bgr']=df.bgr.fillna(value=df['bgr'].mean())
# df['bu']=df.bu.fillna(value=df['bu'].mean())
# df['sc']=df.sc.fillna(value=df['sc'].mean())
# df['sod']=df.sod.fillna(value=df['sod'].mean())
# df['pot']=df.pot.fillna(value=df['pot'].mean())
# df['hemo']=df.hemo.fillna(value=df['hemo'].mean())
# df['rc']=df.rc.fillna(value=df['rc'].mean())
# df['wc']=df.wc.fillna(value=df['wc'].mean())
# df['pcv']=df.pcv.fillna(value=df['pcv'].mean())
# df['sg']=df.sg.fillna(value=df['sg'].mean())

# st.write('''
# ## **WITHOUIT NULL VALUES**''')

# st.write(df.isnull().sum()* 100/len(df))




# #CHANGE CLASSIFICATION "dtype"

# df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
# st.write(df.dtypes)



# #SLIDEBAR ICONS SHOW

# def user_input():
#     age = st.sidebar.slider("age", 1, 90, 2)
#     bp = st.sidebar.slider("blood pressure", 40, 190, 50)
#     sg = st.sidebar.slider("specific gravity", 1.0, 1.025, 1.005)
#     al = st.sidebar.slider("albumin", 0, 5, 0)
#     su = st.sidebar.slider("sugar", 0, 5, 0)
#     bgr = st.sidebar.slider("blood glucose random", 20, 500, 22)
#     bu = st.sidebar.slider("blood urea", 1.0, 400.0, 1.5)
#     sc = st.sidebar.slider("serum creatinine", 0.0, 80.0, 0.4)
#     sod = st.sidebar.slider("sodium", 4.0, 165.0, 4.5)
#     pot = st.sidebar.slider("potassium", 2.0, 50.0, 2.5)
#     hemo = st.sidebar.slider("hemoglobin", 3.0, 18.0, 3.1)
#     pcv = st.sidebar.slider("packed cell volume", 5, 55, 9)
#     wc = st.sidebar.slider("white blood cell count", 2000, 26500, 2200)
#     rc = st.sidebar.slider("red blood cell count", 2.0, 10.0, 2.1)


#     user_input_data = {
#         "age" : age,
#         "blood pressure" : bp,
#         "specific gravity" : sg,
#         "albumin" :al,
#         "sugar" : su,
#         "blood glucose random" : bgr,
#         "blood urea" : bu,
#         "serum creatinine" : sc,
#         "sodium" : sod,
#         "potassium" : pot,
#         "hemoglobin" : hemo,
#         "packed cell volume" : pcv,
#         "white blood cell count" : wc,
#         "red blood cell count" : rc,
#     }
        
#     report_data = pd.DataFrame(user_input_data, index=[0])
#     return report_data

# user_data = user_input()
# st.subheader("Patient's Data")
# st.write(user_data)



# df =df.dropna()
# st.write(df.head(10))


# #ML WORK

# X = df.drop('classification',axis=1)
# y = df['classification']




# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)





# # Define and train the classifiers
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)

# # svmm = svm()
# # svmm.fit(X_train, y_train)

# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# gb = GradientBoostingClassifier()
# gb.fit(X_train, y_train)

# LR = LogisticRegression()
# LR.fit(X_train, y_train)

# # Evaluate the classifiers on the test data
# knn_pred = knn.predict(X_test)
# dt_pred = dt.predict(X_test)
# rf_pred = rf.predict(X_test)
# gb_pred = gb.predict(X_test)
# LR_pred = LR.predict(X_test)


# knn_acc = accuracy_score(y_test, knn_pred)
# dt_acc = accuracy_score(y_test, dt_pred)
# rf_acc = accuracy_score(y_test, rf_pred)
# gb_acc = accuracy_score(y_test, gb_pred)
# LR_acc = accuracy_score(y_test, LR_pred)

# st.write('''
# ## **Accuracy Score**''')

# st.write(knn_acc)



# ============================
# 📦 Import Required Libraries
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# ============================
# ⚙️ Streamlit Config
# ============================
st.set_page_config(page_title="Kidney Disease Prediction", layout="wide")

st.title("🩺 Kidney Disease Prediction App")
st.write("This web app uses multiple ML models to predict kidney disease from patient health indicators.")

# ============================
# 🖼️ Header Image (optional)
# ============================
try:
    image = Image.open("kidney_1.png")
    st.image(image, caption="Kidney Disease Prediction App", use_container_width=True)
except:
    st.info("🖼️ No image found — skipping.")

# ============================
# 📂 Load and Clean Data
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("kidney.csv")
    return df

df = load_data()

st.subheader("📋 Raw Dataset Preview")
st.dataframe(df.head())

# ============================
# 🧹 Data Preprocessing
# ============================
st.header("🧹 Data Cleaning & Encoding")

if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Clean up whitespace and empty strings
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

# Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

st.success("✅ Missing values handled and categorical features encoded.")

# ============================
# 📊 Data Visualization
# ============================
st.header("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("### 🔗 Correlation Heatmap (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm')
    st.pyplot(fig)

with col2:
    st.write("### 📈 Classification Distribution")
    if 'classification' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(x='classification', data=df, palette='viridis')
        plt.title("CKD vs Non-CKD")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ 'classification' column not found in data.")

# ============================
# 🧠 Model Training
# ============================
st.header("🧠 Model Training and Evaluation")

target_col = 'classification'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
st.dataframe(results_df)

fig3, ax3 = plt.subplots()
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='mako')
st.pyplot(fig3)

# ============================
# ⚙️ Sidebar Controls
# ============================
st.sidebar.header("⚙️ Model & Patient Input")

# Model selector
classifier_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
model = models[classifier_name]
model.fit(X_train, y_train)
st.sidebar.success(f"✅ {classifier_name} selected")

# Patient input
st.sidebar.subheader("👤 Patient Data Input")

def sidebar_input_features():
    user_inputs = {}
    for col in X.columns:
        cmin, cmax, cmean = float(df[col].min()), float(df[col].max()), float(df[col].mean())
        user_inputs[col] = st.sidebar.slider(f"{col}", cmin, cmax, cmean)
    return pd.DataFrame([user_inputs])

user_df = sidebar_input_features()

# ============================
# 🔮 Prediction Button
# ============================
predict_btn = st.sidebar.button("🔮 Predict Kidney Disease")

st.subheader("🧾 Patient Input Data")
st.dataframe(user_df)

if predict_btn:
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(user_df)[0]
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("🟥 Chronic Kidney Disease Detected")
    else:
        st.success("🟩 No Kidney Disease Detected")

