import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Machine Learning Study Plan Dashboard", layout="wide")

# Title and header
st.title("Machine Learning Study Plan Dashboard")
st.header("Track Your 48-Week ML Journey")

# Task and deliverable data with new dates starting June 2, 2025
tasks = {
    "Week 1": [
        {"Day": "Jun 2, 2025 (Mon)", "Tasks": ["Install Anaconda", "Open Jupyter Notebook", "Write and run: print('Hello, World!')", "Explore Jupyter: run cells, add markdown"], "Deliverable": "A notebook with your first code"},
        {"Day": "Jun 3, 2025 (Tue)", "Tasks": ["Define variables: x = 5, y = 3.14, name = 'Alex'", "Perform operations: x + y, name * 2", "Check types: type(x), type(name)", "Convert types: int(y), str(x)"], "Deliverable": "Notebook with 5 variable examples"},
        {"Day": "Jun 4, 2025 (Wed)", "Tasks": ["Create a list: numbers = [1, 2, 3, 4]", "Access elements: numbers[0], numbers[-1]", "Use a for loop: for n in numbers: print(n)", "Write an if-else: if x > 0: print('Positive')"], "Deliverable": "Loop printing list items"},
        {"Day": "Jun 5, 2025 (Thu)", "Tasks": ["Define a function: def add(a, b): return a + b", "Call it: add(3, 4)", "Import a module: import math; math.sqrt(16)", "Write a function with if-else (e.g., is_even)"], "Deliverable": "Notebook with 2 custom functions"},
        {"Day": "Jun 6, 2025 (Fri)", "Tasks": ["Write a function to compute factorial", "Use a loop to print numbers 1–10", "Solve: Sum of a list [1, 2, 3]", "Debug any errors from Days 1–4"], "Deliverable": "Notebook with factorial function"}
    ],
    "Week 2": [
        {"Day": "Jun 9, 2025 (Mon)", "Tasks": ["Create a dict: student = {'name': 'Alex', 'age': 20}", "Access values: student['name']", "Add key-value: student['grade'] = 'A'", "Use list comprehension: [x**2 for x in range(5)]"], "Deliverable": "Dict with 3 entries"},
        {"Day": "Jun 10, 2025 (Tue)", "Tasks": ["Install NumPy: pip install numpy", "Create array: import numpy as np; a = np.array([1, 2, 3])", "Access: a[1], slice: a[0:2]", "Basic ops: a + 2, a * 3"], "Deliverable": "Notebook with array ops"},
        {"Day": "Jun 11, 2025 (Wed)", "Tasks": ["Compute: np.sum(a), np.mean(a)", "Create 2D array: b = np.array([[1, 2], [3, 4]])", "Matrix ops: b.T, np.dot(b, b)", "Generate array: np.arange(0, 10, 2)"], "Deliverable": "2D array with sum/mean"},
        {"Day": "Jun 12, 2025 (Thu)", "Tasks": ["Install Pandas: pip install pandas", "Load CSV: pd.read_csv('sample.csv')", "View: df.head(), df.info()", "Select column: df['column_name']"], "Deliverable": "Notebook with loaded DataFrame"},
        {"Day": "Jun 13, 2025 (Fri)", "Tasks": ["Filter DataFrame: df[df['age'] > 25]", "Add column: df['new_col'] = df['old_col'] * 2", "Handle NaN: df.fillna(0)", "Compute stats: df.describe()"], "Deliverable": "Filtered DataFrame"}
    ],
    "Week 3": [
        {"Day": "Jun 16, 2025 (Mon)", "Tasks": ["Install: pip install matplotlib", "Plot: plt.plot([1, 2, 3], [4, 5, 6])", "Add labels: plt.xlabel('X'), plt.title('Line')", "Show: plt.show()"], "Deliverable": "Simple line plot"},
        {"Day": "Jun 17, 2025 (Tue)", "Tasks": ["Install: pip install seaborn", "Load dataset: sns.load_dataset('tips')", "Box plot: sns.boxplot(x='day', y='total_bill', data=df)", "Heatmap: sns.heatmap(df.corr())"], "Deliverable": "Box plot notebook"},
        {"Day": "Jun 18, 2025 (Wed)", "Tasks": ["Download Titanic dataset (Kaggle)", "Load: df = pd.read_csv('titanic.csv')", "Clean: Drop NaN in Age", "Explore: df['Survived'].value_counts()"], "Deliverable": "Cleaned dataset"},
        {"Day": "Jun 19, 2025 (Thu)", "Tasks": ["Plot survival: sns.countplot(x='Survived', data=df)", "Plot by class: sns.catplot(x='Pclass', y='Survived', kind='bar', data=df)", "Save plot: plt.savefig('survival.png')", "Interpret trends"], "Deliverable": "2 plots"},
        {"Day": "Jun 20, 2025 (Fri)", "Tasks": ["Fix errors from Days 1–4", "Add titles/labels to plots", "Summarize findings in markdown", "Save notebook as Foundations.ipynb"], "Deliverable": "Polished notebook"}
    ],
    "Week 4": [
        {"Day": "Jun 23, 2025 (Mon)", "Tasks": ["Read: Supervised learning basics", "Understand: y = mx + b", "Install scikit-learn: pip install scikit-learn", "Load toy data: X = [[1], [2], [3]]; y = [2, 4, 6]"], "Deliverable": "Notes on supervised learning"},
        {"Day": "Jun 24, 2025 (Tue)", "Tasks": ["Import: from sklearn.linear_model import LinearRegression", "Fit: model.fit(X, y)", "Predict: model.predict([[4]])", "Check: model.coef_, model.intercept_"], "Deliverable": "Notebook with predictions"},
        {"Day": "Jun 25, 2025 (Wed)", "Tasks": ["Split data: train_test_split(X, y, test_size=0.2)", "Score: model.score(X_test, y_test)", "Compute MSE: from sklearn.metrics import mean_squared_error"], "Deliverable": "MSE value"},
        {"Day": "Jun 26, 2025 (Thu)", "Tasks": ["Download house price data (Kaggle)", "Fit model: Use Rooms to predict Price", "Predict: New house with 3 rooms", "Plot: plt.scatter(X, y); plt.plot(X, model.predict(X))"], "Deliverable": "Prediction plot"},
        {"Day": "Jun 27, 2025 (Fri)", "Tasks": ["Check R²: model.score(X, y)", "Adjust data: scale with StandardScaler", "Fix errors from Days 1–4", "Save: Supervised1.ipynb"], "Deliverable": "Improved model"}
    ],
    "Week 5": [
        {"Day": "Jun 30, 2025 (Mon)", "Tasks": ["Read: Logistic vs. linear", "Code sigmoid: 1 / (1 + np.exp(-x))", "Load binary data: X = [[1], [2], [3]]; y = [0, 0, 1]", "Import: LogisticRegression"], "Deliverable": "Sigmoid plot"},
        {"Day": "Jul 1, 2025 (Tue)", "Tasks": ["Fit: model.fit(X, y)", "Predict: model.predict([[2.5]])", "Probabilities: model.predict_proba([[2.5]])", "Check: model.coef_"], "Deliverable": "Prediction output"},
        {"Day": "Jul 2, 2025 (Wed)", "Tasks": ["Split data as before", "Accuracy: model.score(X_test, y_test)", "Confusion matrix: confusion_matrix(y_test, model.predict(X_test))"], "Deliverable": "Confusion matrix"},
        {"Day": "Jul 3, 2025 (Thu)", "Tasks": ["Use spam dataset (Kaggle)", "Fit logistic model: Predict spam (1/0)", "Evaluate: Precision, recall", "Plot ROC curve (optional)"], "Deliverable": "Spam prediction"},
        {"Day": "Jul 4, 2025 (Fri)", "Tasks": ["Compare linear vs. logistic", "Note differences in output", "Fix errors from Days 1–4", "Save: Supervised2.ipynb"], "Deliverable": "Comparison notes"}
    ],
    "Week 6": [
        {"Day": "Jul 7, 2025 (Mon)", "Tasks": ["Read: Decision tree basics", "Import: DecisionTreeClassifier", "Fit: tree.fit(X, y)", "Predict: tree.predict([[2]])"], "Deliverable": "Tree prediction"},
        {"Day": "Jul 8, 2025 (Tue)", "Tasks": ["Split data, fit tree", "Score: tree.score(X_test, y_test)", "Limit depth: DecisionTreeClassifier(max_depth=3)", "Visualize (optional): plot_tree"], "Deliverable": "Accuracy score"},
        {"Day": "Jul 9, 2025 (Wed)", "Tasks": ["Import: RandomForestClassifier", "Fit: rf.fit(X, y, n_estimators=10)", "Predict and score", "Check: rf.feature_importances_"], "Deliverable": "RF prediction"},
        {"Day": "Jul 10, 2025 (Thu)", "Tasks": ["Use credit risk dataset (Kaggle)", "Fit RF: Predict default (1/0)", "Evaluate: Accuracy, confusion matrix", "Tweak: Change n_estimators"], "Deliverable": "RF results"},
        {"Day": "Jul 11, 2025 (Fri)", "Tasks": ["Compare tree vs. RF", "Note overfitting signs", "Fix errors", "Save: Supervised3.ipynb"], "Deliverable": "Comparison table"}
    ],
    "Week 7": [
        {"Day": "Jul 14, 2025 (Mon)", "Tasks": ["Read: SVM basics", "Import: SVC", "Fit: svm.fit(X, y, kernel='linear')", "Predict: svm.predict([[2]])"], "Deliverable": "SVM prediction"},
        {"Day": "Jul 15, 2025 (Tue)", "Tasks": ["Try RBF: SVC(kernel='rbf')", "Use iris dataset: load_iris()", "Fit and predict on 2 features", "Score: svm.score(X_test, y_test)"], "Deliverable": "Iris prediction"},
        {"Day": "Jul 16, 2025 (Wed)", "Tasks": ["Pick dataset (e.g., diabetes)", "Load and clean: Drop NaN, scale features", "Split: Train/test sets", "Plan: Test LR, RF, SVM"], "Deliverable": "Cleaned data"},
        {"Day": "Jul 17, 2025 (Thu)", "Tasks": ["Fit 3 models: LR, RF, SVM", "Predict and score each", "Compare accuracy in a table", "Plot: Bar chart of scores"], "Deliverable": "Model comparison"},
        {"Day": "Jul 18, 2025 (Fri)", "Tasks": ["Fix errors, improve best model", "Write: Which model won?", "Save: SupervisedProject.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 8": [
        {"Day": "Jul 21, 2025 (Mon)", "Tasks": ["Import: KMeans", "Load toy data: X = [[1, 2], [2, 3], [5, 6]]", "Fit: kmeans.fit(X, n_clusters=2)", "Labels: kmeans.labels_"], "Deliverable": "Cluster labels"},
        {"Day": "Jul 22, 2025 (Tue)", "Tasks": ["Run kmeans for k=1 to 5", "Compute inertia: kmeans.inertia_", "Plot: Inertia vs. k", "Pick optimal k"], "Deliverable": "Elbow plot"},
        {"Day": "Jul 23, 2025 (Wed)", "Tasks": ["Import: AgglomerativeClustering", "Fit: hc.fit(X, n_clusters=2)", "Labels: hc.labels_", "Plot dendrogram (optional)"], "Deliverable": "HC labels"},
        {"Day": "Jul 24, 2025 (Thu)", "Tasks": ["Use customer data (Kaggle)", "Apply k-means: Cluster by spend/income", "Visualize: Scatter plot of clusters", "Interpret: Describe clusters"], "Deliverable": "Cluster plot"},
        {"Day": "Jul 25, 2025 (Fri)", "Tasks": ["Run k-means and HC on same data", "Compare labels visually", "Fix errors", "Save: Unsupervised1.ipynb"], "Deliverable": "Comparison notes"}
    ],
    "Week 9": [
        {"Day": "Jul 28, 2025 (Mon)", "Tasks": ["Import: DBSCAN", "Fit: db.fit(X, eps=0.5, min_samples=2)", "Labels: db.labels_", "Tweak eps/min_samples"], "Deliverable": "DBSCAN labels"},
        {"Day": "Jul 29, 2025 (Tue)", "Tasks": ["Import: silhouette_score", "Compute: silhouette_score(X, kmeans.labels_)", "Test on k-means and DBSCAN", "Note: Higher score = better clusters"], "Deliverable": "Silhouette scores"},
        {"Day": "Jul 30, 2025 (Wed)", "Tasks": ["Use noisy data (e.g., moons)", "Cluster with k-means and DBSCAN", "Plot: Compare results", "Interpret: Which worked better?"], "Deliverable": "Cluster comparison"},
        {"Day": "Jul 31, 2025 (Thu)", "Tasks": ["Re-run best method on customer data", "Adjust parameters for better score", "Save: Unsupervised2.ipynb"], "Deliverable": "Optimized clusters"},
        {"Day": "Aug 1, 2025 (Fri)", "Tasks": ["Write: Pros/cons of k-means, HC, DBSCAN", "Fix errors", "Save final notebook"], "Deliverable": "Summary markdown"}
    ],
    "Week 10": [
        {"Day": "Aug 4, 2025 (Mon)", "Tasks": ["Import: PCA", "Fit: pca.fit(X, n_components=2)", "Transform: X_pca = pca.transform(X)", "Check: pca.explained_variance_ratio_"], "Deliverable": "Reduced data"},
        {"Day": "Aug 5, 2025 (Tue)", "Tasks": ["Import: TSNE", "Fit: tsne.fit_transform(X, n_components=2)", "Plot: Scatter of X_tsne", "Use MNIST (small subset)"], "Deliverable": "t-SNE plot"},
        {"Day": "Aug 6, 2025 (Wed)", "Tasks": ["Pick dataset (e.g., gene expression)", "Reduce with PCA to 2D", "Cluster with k-means", "Plot: Clusters in 2D"], "Deliverable": "PCA + clusters"},
        {"Day": "Aug 7, 2025 (Thu)", "Tasks": ["Evaluate: Silhouette score", "Compare: Raw vs. PCA-clustered", "Adjust n_components or k", "Plot final clusters"], "Deliverable": "Evaluation table"},
        {"Day": "Aug 8, 2025 (Fri)", "Tasks": ["Write: When to use PCA vs. t-SNE?", "Fix errors", "Save: UnsupervisedProject.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 11": [
        {"Day": "Aug 11, 2025 (Mon)", "Tasks": ["Read: Neuron basics", "Code sigmoid: def sigmoid(x): return 1 / (1 + np.exp(-x))", "Plot sigmoid"], "Deliverable": "Sigmoid plot"},
        {"Day": "Aug 12, 2025 (Tue)", "Tasks": ["Install TensorFlow: pip install tensorflow", "Build NN: Sequential()", "Add layer: Dense(1, input_dim=1)", "Predict: model.predict([[2]])"], "Deliverable": "NN prediction"},
        {"Day": "Aug 13, 2025 (Wed)", "Tasks": ["Compile: model.compile(optimizer='sgd', loss='mse')", "Fit: model.fit(X, y, epochs=10)", "Check weights", "Predict again"], "Deliverable": "Trained model"},
        {"Day": "Aug 14, 2025 (Thu)", "Tasks": ["Build NN for XOR: X = [[0,0], [0,1], [1,0], [1,1]]; y = [0,1,1,0]", "Add hidden layer: Dense(4, activation='relu')", "Fit and predict"], "Deliverable": "XOR predictions"},
        {"Day": "Aug 15, 2025 (Fri)", "Tasks": ["Plot loss: history.history['loss']", "Fix errors, adjust epochs", "Save: DeepLearning1.ipynb"], "Deliverable": "Loss plot"}
    ],
    "Week 12": [
        {"Day": "Aug 18, 2025 (Mon)", "Tasks": ["Try Adam optimizer", "Fit on XOR again", "Compare speed vs. SGD", "Check loss"], "Deliverable": "Adam results"},
        {"Day": "Aug 19, 2025 (Tue)", "Tasks": ["Add dropout: Dropout(0.2)", "Fit with dropout", "Compare overfitting", "Check weights"], "Deliverable": "Dropout model"},
        {"Day": "Aug 20, 2025 (Wed)", "Tasks": ["Use MNIST: mnist.load_data()", "Flatten: X_train.reshape(60000, 784)", "Build NN: 2 layers, fit"], "Deliverable": "MNIST model"},
        {"Day": "Aug 21, 2025 (Thu)", "Tasks": ["Predict: model.predict(X_test)", "Accuracy: model.evaluate(X_test, y_test)", "Plot: Sample digit vs. prediction"], "Deliverable": "Accuracy score"},
        {"Day": "Aug 22, 2025 (Fri)", "Tasks": ["Adjust layers: add 1 more", "Change epochs or batch_size", "Save: DeepLearning2.ipynb"], "Deliverable": "Improved model"}
    ],
    "Week 13": [
        {"Day": "Aug 25, 2025 (Mon)", "Tasks": ["Read: Convolution basics", "Add Conv2D layer: Conv2D(32, (3, 3), activation='relu')", "Reshape MNIST: X_train.reshape(60000, 28, 28, 1)", "Compile model"], "Deliverable": "CNN setup"},
        {"Day": "Aug 26, 2025 (Tue)", "Tasks": ["Add MaxPooling2D((2, 2))", "Add Dense layer: Dense(10, activation='softmax')", "Fit on MNIST", "Check: model.summary()"], "Deliverable": "Trained CNN"},
        {"Day": "Aug 27, 2025 (Wed)", "Tasks": ["Load CIFAR-10: cifar10.load_data()", "Build CNN: 2 conv, 2 pool, 2 dense", "Fit: 5 epochs", "Predict: Sample image"], "Deliverable": "CIFAR prediction"},
        {"Day": "Aug 28, 2025 (Thu)", "Tasks": ["Accuracy: model.evaluate(X_test, y_test)", "Plot: Predicted vs. actual images", "Note: Conv layer effects"], "Deliverable": "Accuracy plot"},
        {"Day": "Aug 29, 2025 (Fri)", "Tasks": ["Tweak filters: 64 instead of 32", "Re-run, compare accuracy", "Save: DeepLearning3.ipynb"], "Deliverable": "Final CNN"}
    ],
    "Week 14": [
        {"Day": "Sep 1, 2025 (Mon)", "Tasks": ["Reload MNIST", "Plan: Build CNN for digits", "Normalize: X_train / 255.0", "Define model: 2 conv, 1 pool"], "Deliverable": "Normalized data"},
        {"Day": "Sep 2, 2025 (Tue)", "Tasks": ["Add layers: Conv, pool, dense", "Compile: loss='sparse_categorical_crossentropy'", "Fit: 5 epochs", "Check: model.summary()"], "Deliverable": "Model structure"},
        {"Day": "Sep 3, 2025 (Wed)", "Tasks": ["Fit with validation: validation_split=0.2", "Plot: Training vs. validation loss", "Adjust epochs if needed"], "Deliverable": "Loss plot"},
        {"Day": "Sep 4, 2025 (Thu)", "Tasks": ["Evaluate: model.evaluate(X_test, y_test)", "Predict: 5 test images", "Plot: Predictions vs. actual"], "Deliverable": "Test accuracy"},
        {"Day": "Sep 5, 2025 (Fri)", "Tasks": ["Save model: model.save('mnist_cnn.h5')", "Write: CNN benefits", "Save: CNNProject.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 15": [
        {"Day": "Sep 8, 2025 (Mon)", "Tasks": ["Read: RNN basics", "Import: SimpleRNN", "Build: model.add(SimpleRNN(10, input_shape=(None, 1)))", "Toy data: X = [[[1], [2], [3]]]"], "Deliverable": "RNN setup"},
        {"Day": "Sep 9, 2025 (Tue)", "Tasks": ["Use LSTM: model.add(LSTM(10, input_shape=(3, 1)))", "Fit on toy sequence", "Predict next step"], "Deliverable": "LSTM prediction"},
        {"Day": "Sep 10, 2025 (Wed)", "Tasks": ["Create data: t = np.arange(0, 100); y = np.sin(t)", "Reshape: X = y[:-1].reshape(1, 99, 1)", "Fit LSTM", "Predict: Next value"], "Deliverable": "Sine prediction"},
        {"Day": "Sep 11, 2025 (Thu)", "Tasks": ["Plot: Predicted vs. actual sine", "Compute MSE manually", "Adjust units or epochs"], "Deliverable": "Prediction plot"},
        {"Day": "Sep 12, 2025 (Fri)", "Tasks": ["Compare RNN vs. LSTM on sine", "Note vanishing gradient", "Save: DeepLearning4.ipynb"], "Deliverable": "Comparison notes"}
    ],
    "Week 16": [
        {"Day": "Sep 15, 2025 (Mon)", "Tasks": ["Read: GAN basics", "Build generator: Dense(10, activation='relu')", "Build discriminator: Dense with sigmoid", "Compile both"], "Deliverable": "GAN setup"},
        {"Day": "Sep 16, 2025 (Tue)", "Tasks": ["Install: pip install transformers", "Load: pipeline('sentiment-analysis')", "Test: classifier('I love AI')"], "Deliverable": "Sentiment output"},
        {"Day": "Sep 17, 2025 (Wed)", "Tasks": ["Classify 5 sentences with transformer", "Note: Positive/negative scores", "Try text generation"], "Deliverable": "Classification results"},
        {"Day": "Sep 18, 2025 (Thu)", "Tasks": ["Compare transformer vs. LSTM on text", "Plot: Sentiment scores", "Fix errors"], "Deliverable": "Comparison plot"},
        {"Day": "Sep 19, 2025 (Fri)", "Tasks": ["Write: GAN vs. Transformer uses", "Save: DeepLearning5.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 17": [
        {"Day": "Sep 22, 2025 (Mon)", "Tasks": ["Install: pip install nltk", "Tokenize: nltk.word_tokenize('I love AI')", "Clean: Remove punctuation"], "Deliverable": "Token list"},
        {"Day": "Sep 23, 2025 (Tue)", "Tasks": ["Download: nltk.download('stopwords')", "Remove stopwords", "Stem: PorterStemmer().stem('running')"], "Deliverable": "Cleaned tokens"},
        {"Day": "Sep 24, 2025 (Wed)", "Tasks": ["Import: CountVectorizer", "Fit: vectorizer.fit_transform(['I love AI', 'AI is great'])", "Check: X.toarray()"], "Deliverable": "BoW matrix"},
        {"Day": "Sep 25, 2025 (Thu)", "Tasks": ["Use spam dataset", "Vectorize text, fit logistic regression", "Predict: Spam or not", "Score accuracy"], "Deliverable": "Spam prediction"},
        {"Day": "Sep 26, 2025 (Fri)", "Tasks": ["Plot: Top 10 words", "Fix errors", "Save: NLP1.ipynb"], "Deliverable": "Word frequency plot"}
    ],
    "Week 18": [
        {"Day": "Sep 29, 2025 (Mon)", "Tasks": ["Use VADER: SentimentIntensityAnalyzer", "Score: sia.polarity_scores('I love AI')", "Test 5 sentences"], "Deliverable": "Sentiment scores"},
        {"Day": "Sep 30, 2025 (Tue)", "Tasks": ["Use transformer pipeline", "Classify: classifier('I hate bugs')", "Compare with VADER"], "Deliverable": "BERT vs. VADER"},
        {"Day": "Oct 1, 2025 (Wed)", "Tasks": ["Use: generator = pipeline('text-generation')", "Generate: generator('AI will', max_length=10)", "Test 3 prompts"], "Deliverable": "Generated text"},
        {"Day": "Oct 2, 2025 (Thu)", "Tasks": ["Generate 5 sentences", "Classify sentiment of each", "Plot: Sentiment distribution"], "Deliverable": "Sentiment plot"},
        {"Day": "Oct 3, 2025 (Fri)", "Tasks": ["Fix errors, improve generation", "Save: NLP2.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 19": [
        {"Day": "Oct 6, 2025 (Mon)", "Tasks": ["Pick dataset (e.g., tweets)", "Load and clean: Tokenize, remove stopwords", "Plan: Sentiment or classification"], "Deliverable": "Cleaned data"},
        {"Day": "Oct 7, 2025 (Tue)", "Tasks": ["Vectorize: Use CountVectorizer or TF-IDF", "Split: Train/test sets", "Check shape: X_train.shape"], "Deliverable": "Vectorized data"},
        {"Day": "Oct 8, 2025 (Wed)", "Tasks": ["Fit logistic regression or transformer", "Predict: Test set labels", "Score: Accuracy"], "Deliverable": "Model output"},
        {"Day": "Oct 9, 2025 (Thu)", "Tasks": ["Evaluate: Confusion matrix", "Plot: Predicted vs. actual", "Adjust model if needed"], "Deliverable": "Evaluation plot"},
        {"Day": "Oct 10, 2025 (Fri)", "Tasks": ["Write: Project summary", "Save: NLPProject.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 20": [
        {"Day": "Oct 13, 2025 (Mon)", "Tasks": ["Read: Image classification basics", "Load MNIST", "Use: ResNet50", "Plan: Fine-tune"], "Deliverable": "Notes"},
        {"Day": "Oct 14, 2025 (Tue)", "Tasks": ["Load ResNet: weights='imagenet'", "Add layer: Dense(10, activation='softmax')", "Freeze layers", "Compile"], "Deliverable": "Model setup"},
        {"Day": "Oct 15, 2025 (Wed)", "Tasks": ["Reshape MNIST for RGB", "Fit: 3 epochs", "Evaluate: Accuracy"], "Deliverable": "Trained model"},
        {"Day": "Oct 16, 2025 (Thu)", "Tasks": ["Use cats/dogs dataset (Kaggle)", "Fine-tune ResNet", "Predict: 5 images", "Score accuracy"], "Deliverable": "Predictions"},
        {"Day": "Oct 17, 2025 (Fri)", "Tasks": ["Plot: Predicted vs. actual", "Fix errors", "Save: Vision1.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 21": [
        {"Day": "Oct 20, 2025 (Mon)", "Tasks": ["Read: YOLO basics", "Install: pip install opencv-python", "Load pre-trained YOLO", "Test: Detect in sample image"], "Deliverable": "Detection output"},
        {"Day": "Oct 21, 2025 (Tue)", "Tasks": ["Use own image with YOLO", "Draw boxes: cv2.rectangle()", "Save: Detected image"], "Deliverable": "Annotated image"},
        {"Day": "Oct 22, 2025 (Wed)", "Tasks": ["Read: U-Net basics", "Load pre-trained U-Net", "Test: Segment sample image"], "Deliverable": "Segmented output"},
        {"Day": "Oct 23, 2025 (Thu)", "Tasks": ["Segment road image (Kaggle)", "Compare: Raw vs. segmented", "Adjust threshold if needed"], "Deliverable": "Segmentation plot"},
        {"Day": "Oct 24, 2025 (Fri)", "Tasks": ["Write: Detection vs. segmentation uses", "Save: Vision2.ipynb"], "Deliverable": "Comparison notes"}
    ],
    "Week 22": [
        {"Day": "Oct 27, 2025 (Mon)", "Tasks": ["Pick task: Classify or detect", "Load dataset", "Plan: Use ResNet or YOLO"], "Deliverable": "Dataset ready"},
        {"Day": "Oct 28, 2025 (Tue)", "Tasks": ["Build model: Fine-tune ResNet or load YOLO", "Preprocess: Resize/normalize images", "Fit or detect"], "Deliverable": "Model output"},
        {"Day": "Oct 29, 2025 (Wed)", "Tasks": ["Predict: 10 test images", "Evaluate: Accuracy or IoU", "Plot: Results"], "Deliverable": "Test plot"},
        {"Day": "Oct 30, 2025 (Thu)", "Tasks": ["Adjust model (e.g., epochs)", "Re-test, improve score", "Fix errors"], "Deliverable": "Improved results"},
        {"Day": "Oct 31, 2025 (Fri)", "Tasks": ["Save: VisionProject.ipynb", "Write: Project summary"], "Deliverable": "Final notebook"}
    ],
    "Week 23": [
        {"Day": "Nov 3, 2025 (Mon)", "Tasks": ["Read: RL basics", "Install: pip install gym", "Load: env = gym.make('CartPole-v1')", "Step: env.step()"], "Deliverable": "Gym test run"},
        {"Day": "Nov 4, 2025 (Tue)", "Tasks": ["Define grid-world: 2x2 grid", "Code value iteration", "Find policy: Best actions"], "Deliverable": "Policy table"},
        {"Day": "Nov 5, 2025 (Wed)", "Tasks": ["Run CartPole: Random actions", "Record: env.render()", "Compute: Average reward"], "Deliverable": "Reward score"},
        {"Day": "Nov 6, 2025 (Thu)", "Tasks": ["Read: Monte Carlo basics", "Simulate: 10 episodes in CartPole", "Average returns per action"], "Deliverable": "MC returns"},
        {"Day": "Nov 7, 2025 (Fri)", "Tasks": ["Compare DP vs. MC", "Fix errors", "Save: RL1.ipynb"], "Deliverable": "Comparison notes"}
    ],
    "Week 24": [
        {"Day": "Nov 10, 2025 (Mon)", "Tasks": ["Define Q-table: q_table = np.zeros((state_size, action_size))", "Update Q-table", "Test on toy grid"], "Deliverable": "Q-table"},
        {"Day": "Nov 11, 2025 (Tue)", "Tasks": ["Apply Q-learning to CartPole", "Discretize states", "Train: 100 episodes"], "Deliverable": "Trained Q-table"},
        {"Day": "Nov 12, 2025 (Wed)", "Tasks": ["Plot: Reward over episodes", "Tweak: Alpha, gamma", "Save: RL2.ipynb"], "Deliverable": "Reward plot"},
        {"Day": "Nov 13, 2025 (Thu)", "Tasks": ["Setup: CarRacing-v0 or CartPole", "Plan: Q-learning or random policy", "Run: Baseline performance"], "Deliverable": "Baseline score"},
        {"Day": "Nov 14, 2025 (Fri)", "Tasks": ["Train: Q-learning on env", "Record: Rewards per episode", "Plot: Training progress"], "Deliverable": "Training plot"}
    ],
    "Week 25": [
        {"Day": "Nov 17, 2025 (Mon)", "Tasks": ["Tweak: Learning rate, epsilon", "Re-train: 50 more episodes", "Compare: New vs. old rewards"], "Deliverable": "Adjusted model"},
        {"Day": "Nov 18, 2025 (Tue)", "Tasks": ["Test: 10 episodes with learned policy", "Average reward", "Plot: Test performance"], "Deliverable": "Test results"},
        {"Day": "Nov 19, 2025 (Wed)", "Tasks": ["Fix errors, optimize policy", "Save: RLProject.ipynb", "Write: Lessons learned"], "Deliverable": "Final notebook"},
        {"Day": "Nov 20, 2025 (Thu)", "Tasks": ["Re-run best policy", "Compare: Random vs. trained"], "Deliverable": "Comparison table"},
        {"Day": "Nov 21, 2025 (Fri)", "Tasks": ["Write: RL strengths/weaknesses", "Save all work"], "Deliverable": "Summary markdown"}
    ],
    "Week 26": [
        {"Day": "Nov 24, 2025 (Mon)", "Tasks": ["Read: AI ethics overview", "Note: 3 ethical concerns (e.g., bias)"], "Deliverable": "Notes"},
        {"Day": "Nov 25, 2025 (Tue)", "Tasks": ["Load dataset (e.g., Titanic)", "Check: Bias in Sex vs. Survived", "Plot: Survival by gender"], "Deliverable": "Bias plot"},
        {"Day": "Nov 26, 2025 (Wed)", "Tasks": ["Read: Fairness metrics", "Compute: Equal opportunity", "Test on Week 7 model"], "Deliverable": "Fairness score"},
        {"Day": "Nov 27, 2025 (Thu)", "Tasks": ["Read: GDPR basics", "List: 3 responsible AI rules", "Apply: To Week 7 project"], "Deliverable": "Rules list"},
        {"Day": "Nov 28, 2025 (Fri)", "Tasks": ["Audit: Week 7 model for bias", "Write: Fixes (e.g., reweight data)", "Save: Ethics.ipynb"], "Deliverable": "Audit report"}
    ],
    "Week 27": [
        {"Day": "Dec 1, 2025 (Mon)", "Tasks": ["Read: Model optimization", "Load: Week 14 CNN", "Check size: model.summary()"], "Deliverable": "Model size notes"},
        {"Day": "Dec 2, 2025 (Tue)", "Tasks": ["Install: pip install tensorflow-model-optimization", "Prune: prune_low_magnitude(model)", "Re-train: 2 epochs"], "Deliverable": "Pruned model"},
        {"Day": "Dec 3, 2025 (Wed)", "Tasks": ["Install: pip install flask", "Code: Basic Flask app", "Test: python app.py"], "Deliverable": "Running app"},
        {"Day": "Dec 4, 2025 (Thu)", "Tasks": ["Deploy CNN: Predict via Flask", "Test: Send image, get prediction", "Fix errors"], "Deliverable": "Deployed endpoint"},
        {"Day": "Dec 5, 2025 (Fri)", "Tasks": ["Compare: Pruned vs. original accuracy", "Save: Optimization.ipynb"], "Deliverable": "Comparison table"}
    ],
    "Week 28": [
        {"Day": "Day 1", "Tasks": ["Read: RecSys basics"], "Deliverable": "Notes"},
        {"Day": "Day 2", "Tasks": ["Load MovieLens (Kaggle)"], "Deliverable": "Loaded dataset"},
        {"Day": "Day 3", "Tasks": ["Collaborative filtering: User-item matrix"], "Deliverable": "User-item matrix"},
        {"Day": "Day 4", "Tasks": ["Compute similarity: cosine_similarity"], "Deliverable": "Similarity scores"},
        {"Day": "Day 5", "Tasks": ["Predict: Top 5 movies for user"], "Deliverable": "Movie predictions"}
    ],
    "Week 29": [
        {"Day": "Day 1", "Tasks": ["Content-based: Extract features"], "Deliverable": "Feature set"},
        {"Day": "Day 2", "Tasks": ["Vectorize movie titles"], "Deliverable": "Vectorized titles"},
        {"Day": "Day 3", "Tasks": ["Build content model"], "Deliverable": "Content model"},
        {"Day": "Day 4", "Tasks": ["Predict: Movies by genre"], "Deliverable": "Genre predictions"},
        {"Day": "Day 5", "Tasks": ["Compare: Collab vs. content"], "Deliverable": "Comparison notes"}
    ],
    "Week 30": [
        {"Day": "Day 1", "Tasks": ["Hybrid: Combine scores"], "Deliverable": "Hybrid model"},
        {"Day": "Day 2", "Tasks": ["Test: 10 users"], "Deliverable": "User predictions"},
        {"Day": "Day 3", "Tasks": ["Plot: Precision/recall"], "Deliverable": "Precision/recall plot"},
        {"Day": "Day 4", "Tasks": ["Adjust weights"], "Deliverable": "Adjusted model"},
        {"Day": "Day 5", "Tasks": ["Save: RecSys1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 31": [
        {"Day": "Day 1", "Tasks": ["Refine: Add more features"], "Deliverable": "Enhanced model"},
        {"Day": "Day 2", "Tasks": ["Test new predictions"], "Deliverable": "New predictions"},
        {"Day": "Day 3", "Tasks": ["Fix errors"], "Deliverable": "Error-free model"},
        {"Day": "Day 4", "Tasks": ["Write: Summary"], "Deliverable": "Summary markdown"},
        {"Day": "Day 5", "Tasks": ["Save: RecSysProject.ipynb"], "Deliverable": "Final notebook"}
    ],
    "Week 32": [
        {"Day": "Day 1", "Tasks": ["Read: Chatbot basics"], "Deliverable": "Notes"},
        {"Day": "Day 2", "Tasks": ["Install: pip install rasa"], "Deliverable": "Rasa installed"},
        {"Day": "Day 3", "Tasks": ["Define intents: nlu.yml"], "Deliverable": "Intents file"},
        {"Day": "Day 4", "Tasks": ["Write stories: stories.yml"], "Deliverable": "Stories file"},
        {"Day": "Day 5", "Tasks": ["Train: rasa train"], "Deliverable": "Trained model"}
    ],
    "Week 33": [
        {"Day": "Day 1", "Tasks": ["Test: rasa shell"], "Deliverable": "Test output"},
        {"Day": "Day 2", "Tasks": ["Add responses: domain.yml"], "Deliverable": "Responses file"},
        {"Day": "Day 3", "Tasks": ["Improve NLU: Add examples"], "Deliverable": "Updated NLU"},
        {"Day": "Day 4", "Tasks": ["Re-train and test"], "Deliverable": "New model"},
        {"Day": "Day 5", "Tasks": ["Save: Chatbot1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 34": [
        {"Day": "Day 1", "Tasks": ["Add custom action"], "Deliverable": "Custom action"},
        {"Day": "Day 2", "Tasks": ["Test: 5 conversations"], "Deliverable": "Conversation logs"},
        {"Day": "Day 3", "Tasks": ["Plot: Intent accuracy"], "Deliverable": "Accuracy plot"},
        {"Day": "Day 4", "Tasks": ["Fix errors"], "Deliverable": "Error-free model"},
        {"Day": "Day 5", "Tasks": ["Refine responses"], "Deliverable": "Improved responses"}
    ],
    "Week 35": [
        {"Day": "Day 1", "Tasks": ["Deploy: rasa run"], "Deliverable": "Running chatbot"},
        {"Day": "Day 2", "Tasks": ["Test via API"], "Deliverable": "API test results"},
        {"Day": "Day 3", "Tasks": ["Write: Chatbot summary"], "Deliverable": "Summary markdown"},
        {"Day": "Day 4", "Tasks": ["Save: ChatbotProject.ipynb"], "Deliverable": "Final notebook"},
        {"Day": "Day 5", "Tasks": ["Review all steps"], "Deliverable": "Review notes"}
    ],
    "Week 36": [
        {"Day": "Day 1", "Tasks": ["Install: pip install carla"], "Deliverable": "CARLA installed"},
        {"Day": "Day 2", "Tasks": ["Setup: Load CARLA env"], "Deliverable": "Environment setup"},
        {"Day": "Day 3", "Tasks": ["Random policy: Drive car"], "Deliverable": "Random drive"},
        {"Day": "Day 4", "Tasks": ["Record: Baseline reward"], "Deliverable": "Baseline score"},
        {"Day": "Day 5", "Tasks": ["Plot: Baseline performance"], "Deliverable": "Performance plot"}
    ],
    "Week 37": [
        {"Day": "Day 1", "Tasks": ["Define Q-table for driving"], "Deliverable": "Q-table"},
        {"Day": "Day 2", "Tasks": ["Train: 50 episodes"], "Deliverable": "Trained model"},
        {"Day": "Day 3", "Tasks": ["Plot: Reward over time"], "Deliverable": "Reward plot"},
        {"Day": "Day 4", "Tasks": ["Adjust: Epsilon decay"], "Deliverable": "Adjusted model"},
        {"Day": "Day 5", "Tasks": ["Re-train: 50 more"], "Deliverable": "Updated model"}
    ],
    "Week 38": [
        {"Day": "Day 1", "Tasks": ["Test: 10 episodes"], "Deliverable": "Test results"},
        {"Day": "Day 2", "Tasks": ["Evaluate: Average reward"], "Deliverable": "Reward score"},
        {"Day": "Day 3", "Tasks": ["Plot: Test vs. train"], "Deliverable": "Comparison plot"},
        {"Day": "Day 4", "Tasks": ["Fix errors"], "Deliverable": "Error-free model"},
        {"Day": "Day 5", "Tasks": ["Save: Driving1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 39": [
        {"Day": "Day 1", "Tasks": ["Refine: Add state features"], "Deliverable": "Enhanced model"},
        {"Day": "Day 2", "Tasks": ["Re-test: Improved policy"], "Deliverable": "Test results"},
        {"Day": "Day 3", "Tasks": ["Write: Summary"], "Deliverable": "Summary markdown"},
        {"Day": "Day 4", "Tasks": ["Save: DrivingProject.ipynb"], "Deliverable": "Final notebook"},
        {"Day": "Day 5", "Tasks": ["Review all steps"], "Deliverable": "Review notes"}
    ],
    "Week 40": [
        {"Day": "Day 1", "Tasks": ["Install: pip install opencv-python deepface"], "Deliverable": "Libraries installed"},
        {"Day": "Day 2", "Tasks": ["Load image: cv2.imread()"], "Deliverable": "Loaded image"},
        {"Day": "Day 3", "Tasks": ["Detect face: cv2.CascadeClassifier"], "Deliverable": "Face detection"},
        {"Day": "Day 4", "Tasks": ["Test: Detect in 5 images"], "Deliverable": "Detected images"},
        {"Day": "Day 5", "Tasks": ["Save: Detected images"], "Deliverable": "Saved images"}
    ],
    "Week 41": [
        {"Day": "Day 1", "Tasks": ["Use DeepFace: DeepFace.verify()"], "Deliverable": "Verification results"},
        {"Day": "Day 2", "Tasks": ["Compare: 2 faces"], "Deliverable": "Comparison results"},
        {"Day": "Day 3", "Tasks": ["Build recognizer: Match database"], "Deliverable": "Recognizer model"},
        {"Day": "Day 4", "Tasks": ["Test: 5 known faces"], "Deliverable": "Match results"},
        {"Day": "Day 5", "Tasks": ["Plot: Match scores"], "Deliverable": "Score plot"}
    ],
    "Week 42": [
        {"Day": "Day 1", "Tasks": ["Add security: Unknown face alert"], "Deliverable": "Alert system"},
        {"Day": "Day 2", "Tasks": ["Test: 5 unknown faces"], "Deliverable": "Test results"},
        {"Day": "Day 3", "Tasks": ["Evaluate: Accuracy"], "Deliverable": "Accuracy score"},
        {"Day": "Day 4", "Tasks": ["Fix errors"], "Deliverable": "Error-free model"},
        {"Day": "Day 5", "Tasks": ["Save: FaceRec1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 43": [
        {"Day": "Day 1", "Tasks": ["Refine: Improve detection"], "Deliverable": "Improved model"},
        {"Day": "Day 2", "Tasks": ["Re-test: Full dataset"], "Deliverable": "Test results"},
        {"Day": "Day 3", "Tasks": ["Write: Summary"], "Deliverable": "Summary markdown"},
        {"Day": "Day 4", "Tasks": ["Save: FaceRecProject.ipynb"], "Deliverable": "Final notebook"},
        {"Day": "Day 5", "Tasks": ["Review all steps"], "Deliverable": "Review notes"}
    ],
    "Week 44": [
        {"Day": "Day 1", "Tasks": ["Read: 1 AI paper (arXiv)"], "Deliverable": "Notes"},
        {"Day": "Day 2", "Tasks": ["Summarize: Key points"], "Deliverable": "Summary"},
        {"Day": "Day 3", "Tasks": ["Read: Quantum ML intro"], "Deliverable": "Notes"},
        {"Day": "Day 4", "Tasks": ["Notes: Future trends"], "Deliverable": "Trend notes"},
        {"Day": "Day 5", "Tasks": ["Save: Trends1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 45": [
        {"Day": "Day 1", "Tasks": ["Read: Federated learning"], "Deliverable": "Notes"},
        {"Day": "Day 2", "Tasks": ["Test: Simple example"], "Deliverable": "Example output"},
        {"Day": "Day 3", "Tasks": ["Compare: Centralized vs. federated"], "Deliverable": "Comparison notes"},
        {"Day": "Day 4", "Tasks": ["Write: Pros/cons"], "Deliverable": "Pros/cons markdown"},
        {"Day": "Day 5", "Tasks": ["Save: Trends2.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 46": [
        {"Day": "Day 1", "Tasks": ["Plan capstone: Pick topic"], "Deliverable": "Topic selection"},
        {"Day": "Day 2", "Tasks": ["Outline: Data, model, goal"], "Deliverable": "Project outline"},
        {"Day": "Day 3", "Tasks": ["Gather: Dataset"], "Deliverable": "Dataset"},
        {"Day": "Day 4", "Tasks": ["Preprocess: Clean data"], "Deliverable": "Cleaned data"},
        {"Day": "Day 5", "Tasks": ["Save: CapstonePlan.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 47": [
        {"Day": "Day 1", "Tasks": ["Build: Base model"], "Deliverable": "Base model"},
        {"Day": "Day 2", "Tasks": ["Train: Initial run"], "Deliverable": "Training results"},
        {"Day": "Day 3", "Tasks": ["Evaluate: Baseline score"], "Deliverable": "Baseline score"},
        {"Day": "Day 4", "Tasks": ["Adjust: Improve model"], "Deliverable": "Improved model"},
        {"Day": "Day 5", "Tasks": ["Save: Capstone1.ipynb"], "Deliverable": "Notebook"}
    ],
    "Week 48": [
        {"Day": "May 4, 2026 (Mon)", "Tasks": ["Test: Full dataset"], "Deliverable": "Test results"},
        {"Day": "May 5, 2026 (Tue)", "Tasks": ["Plot: Final results"], "Deliverable": "Results plot"},
        {"Day": "May 6, 2026 (Wed)", "Tasks": ["Write: Project report"], "Deliverable": "Report"},
        {"Day": "May 7, 2026 (Thu)", "Tasks": ["Build portfolio: GitHub repo"], "Deliverable": "GitHub repo"},
        {"Day": "May 8, 2026 (Fri)", "Tasks": ["Review: All work", "Save: CapstoneFinal.ipynb"], "Deliverable": "Final notebook"}
    ]
}

# Initialize session state for task completion
if 'task_states' not in st.session_state:
    st.session_state.task_states = {week: {task: False for day in tasks[week] for task in day["Tasks"]} for week in tasks}

# Calculate progress for each week
progress = pd.DataFrame({
    "Week": [f"Week {i}" for i in range(1, 49)],
    "Progress": [0.0 for _ in range(1, 49)]
})
for week in tasks:
    total_tasks = sum(len(day["Tasks"]) for day in tasks[week])
    completed_tasks = sum(1 for task in st.session_state.task_states[week].values() if task)
    progress.loc[progress["Week"] == week, "Progress"] = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

# Plotly bar chart for progress
st.subheader("Weekly Progress")
fig = px.bar(
    progress,
    x="Progress",
    y="Week",
    orientation="h",
    title="Progress by Week",
    labels={"Progress": "Completion (%)", "Week": "Week"},
    height=800,
    color="Progress",
    color_continuous_scale="Viridis",
    range_x=[0, 100]
)
fig.update_layout(
    yaxis={"tickfont": {"size": 10}},
    margin=dict(l=100, r=20, t=50, b=20),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Sidebar for week selection
st.sidebar.header("Navigation")
week = st.sidebar.selectbox("Select Week", list(tasks.keys()))

# Display tasks and deliverables for selected week
st.header(f"Tasks for {week}")
for day in tasks[week]:
    with st.expander(day["Day"]):
        st.subheader("Tasks")
        for task in day["Tasks"]:
            st.session_state.task_states[week][task] = st.checkbox(task, value=st.session_state.task_states[week][task], key=f"{week}_{day['Day']}_{task}")
        st.subheader("Deliverable")
        st.write(day["Deliverable"])

# File uploader for deliverables
st.header("Upload Deliverables")
uploaded_file = st.file_uploader(f"Upload deliverable for {week}", type=["ipynb", "png", "pdf"], key=week)
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

# Sample visualization (e.g., Week 8 elbow plot)
if week == "Week 8":
    st.subheader("Sample Visualization: Elbow Plot (Week 8, Day 2)")
    elbow_data = pd.DataFrame({
        "k": range(1, 6),
        "Inertia": [1000, 500, 300, 200, 150]
    })
    fig = px.line(elbow_data, x="k", y="Inertia", markers=True, title="Elbow Method for K-means")
    fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)

# Summary stats
st.header("Summary Stats")
total_days = sum(len(tasks[w]) for w in tasks)
completed_days = sum(len(tasks[w]) for w in tasks if progress[progress["Week"] == w]["Progress"].iloc[0] == 100)
st.metric("Total Days", total_days)
st.metric("Completed Days", completed_days)
st.metric("Completion Rate", f"{(completed_days/total_days)*100:.1f}%")