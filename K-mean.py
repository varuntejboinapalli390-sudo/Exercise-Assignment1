from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

with open("Task2_Data_Clustering.txt", "r") as f:
    text = f.read()

# Clean text step by step
text = text.replace("array([[", "")
text = text.replace("]])", "")
text = text.replace("],", "\n")
text = text.replace("[", "")
text = text.replace("]", "")
text = text.replace(",", " ")

# Now convert
from io import StringIO
data = np.loadtxt(StringIO(text))

df = pd.DataFrame(data, columns=["Feature1", "Feature2", "Feature3", "Feature4"])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

print("Inertia values by k")
for i, val in enumerate(inertia, start=1):
    print(f"k = {i}  ->  inertia = {val}")

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters

print(df.head(10))
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("Centers:")
print(centers_original)
