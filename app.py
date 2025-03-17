import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset with caching
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names

    required_columns = {'name', 'score'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing columns: {required_columns - set(df.columns)}. Ensure your CSV has 'Name' and 'Score'.")
        return None

    df_filtered = df[['name', 'score']].copy()
    df_filtered['marks'] = df_filtered['score'].astype(str).str.split('/').str[0].astype(int)
    df_filtered.rename(columns={'name': 'Name'}, inplace=True)  # Keep only Name, remove marks
    return df_filtered[['Name']]

# Perform K-Means clustering with caching
@st.cache_data
def cluster_students(data, num_clusters=3):
    data['Marks'] = np.random.randint(20, 100, size=len(data))  # Generate random marks for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(data[['Marks']])
    cluster_labels = {0: 'C', 1: 'B', 2: 'A'}  # Assigning A, B, C labels
    data['Category'] = data['Cluster'].map(cluster_labels)
    return data[['Name', 'Category']]  # Keep only Name & Category

# Create segregated groups for all students
def create_grouped_batches(data):
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle students
    grouped_batches = [shuffled_data.iloc[i:i + 5] for i in range(0, len(shuffled_data), 5)]
    return grouped_batches

# Streamlit UI
st.title("Unsupervised Learning: Student Clustering and Batch Formation")

st.write("Upload a CSV file containing 'Name' and 'Score' columns.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Loading data..."):
        data = load_data(uploaded_file)

    if data is not None:
        st.success("Data Loaded Successfully!")
        st.write("### Student Data")  # Updated title
        st.write(data)  # Only Name column

        # Cluster Students
        with st.spinner("Clustering students..."):
            data = cluster_students(data)
        st.success("Clustering Completed!")
        st.write("### Clustered Student Groups")
        st.write(data)  # Only Name & Category

        # Export Categories as List
        categorized_list = data.groupby('Category')['Name'].apply(list).to_dict()
        st.write("### Clustered Groups as List")
        st.write(categorized_list)

        # Create Groups for All Students
        with st.spinner("Creating groups for all students..."):
            grouped_batches = create_grouped_batches(data)
        st.success("Groups Created Successfully!")

        for i, group in enumerate(grouped_batches):
            st.write(f"### Group {i+1}")
            st.write(group[['Name']])  # Show only names

        # User Input for Searching Student's Group and Team Members
        st.write("### Check Student Group and Team Members")
        student_name = st.text_input("Enter Student Name:")
        if student_name:
            student_info = data[data['Name'].str.lower() == student_name.lower()]
            if not student_info.empty:
                student_group = student_info['Category'].values[0]
                st.write(f"**{student_name}** belongs to Group: **{student_group}**")

                # Find and display team members (only names)
                team_members = data[data['Category'] == student_group]
                st.write(f"### Team Members in Group {student_group}")
                st.write(team_members[['Name']])  # Show only names
            else:
                st.warning("Student not found. Please check the spelling.")

        # Visualization
        st.write("### Visualization")
        chart_type = st.radio("Choose visualization type:", ["Bar Chart", "Stacked Bar Chart"])

        fig, ax = plt.subplots(figsize=(8, 5))  # Create figure and axis
        colors = {'A': 'gold', 'B': 'peru', 'C': 'silver'}

        if chart_type == "Bar Chart":
            ax.barh(data['Name'], np.random.randint(20, 100, size=len(data)), 
                    color=[colors.get(cat, 'gray') for cat in data['Category']])
            ax.set_xlabel("Marks")
            ax.set_ylabel("Students")
            ax.set_title("Marks Distribution by Student")

        else:  # Stacked Bar Chart
            grouped_data = data.groupby('Category').size()
            ax.barh(grouped_data.index, grouped_data.values, 
                    color=[colors.get(cat, 'gray') for cat in grouped_data.index])
            ax.set_xlabel("Total Students")
            ax.set_ylabel("Groups")
            ax.set_title("Stacked Representation of Groups")

        st.pyplot(fig, clear_figure=True)

        st.write("Click the button below to regenerate clusters and batches.")
        if st.button("Regenerate Data"):
            st.experimental_rerun()
