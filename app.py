import pickle
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
import threading
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Create data directory if it doesn't exist
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Store training jobs
training_jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files')
def files():
    return render_template('files.html')

@app.route('/upload', methods=['POST'])
def upload():
    ALLOWED_EXTENSIONS = {'csv'}

    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')  # Retrieve multiple files

    uploaded_files = []
    for file in files:
        if file.filename == '':
            continue  # Skip empty filenames
        
        if file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
        else:
            return jsonify({'error': f'File {file.filename} has an invalid extension'}), 400

    return jsonify({'message': 'Files uploaded successfully', 'files': uploaded_files}), 200

@app.route('/api/files')
def get_files():
    files_list = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.csv'):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file_stats = os.stat(file_path)
            files_list.append({
                'name': filename,
                'size': file_stats.st_size,
                'date': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
    
    return jsonify({'files': files_list})

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        os.remove(file_path)
        return jsonify({'message': f'File {filename} deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model Training API
@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        
        if not data or 'model' not in data or 'file' not in data:
            return jsonify({'error': 'Missing model or file parameter'}), 400
        
        model_type = data['model']
        filename = data['file']
        
        # Validate file exists
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': f'File {filename} not found'}), 404
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        training_jobs[job_id] = {
            'status': 'initializing',
            'message': 'Initializing training job...',
            'progress': 0,
            'model': model_type,
            'file': filename
        }
        
        # Start training in a background thread
        training_thread = threading.Thread(
            target=run_training_job,
            args=(job_id, model_type, file_path)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'message': 'Training job started', 'job_id': job_id}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/status/<job_id>')
def get_training_status(job_id):
    if job_id not in training_jobs:
        return jsonify({'error': 'Training job not found'}), 404
    
    return jsonify(training_jobs[job_id]), 200

def run_training_job(job_id, model_type, file_path):
    try:
        training_jobs[job_id]['status'] = 'loading'
        training_jobs[job_id]['message'] = 'Loading dataset...'
        training_jobs[job_id]['progress'] = 5

        df = pd.read_csv(file_path)

        training_jobs[job_id]['status'] = 'preprocessing'
        training_jobs[job_id]['message'] = 'Preprocessing data...'
        training_jobs[job_id]['progress'] = 15

        # Handle missing values
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
        if 'Fare' in df.columns:
            df['Fare'].fillna(df['Fare'].median(), inplace=True)
        if 'Embarked' in df.columns:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

        training_jobs[job_id]['status'] = 'splitting'
        training_jobs[job_id]['message'] = 'Splitting data into train and test sets...'
        training_jobs[job_id]['progress'] = 25

        target_column = 'Survived' if 'Survived' in df.columns else df.columns[0]
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        training_jobs[job_id]['status'] = 'training'
        training_jobs[job_id]['message'] = 'Training model...'
        training_jobs[job_id]['progress'] = 40

        is_classification = len(np.unique(y)) <= 10

        model, metrics, feature_importance = train_selected_model(
            model_type, X_train, y_train, X_test, y_test, is_classification
        )

        training_jobs[job_id]['status'] = 'saving'
        training_jobs[job_id]['message'] = 'Saving trained model...'
        training_jobs[job_id]['progress'] = 90

        # Ensure the models directory exists
        MODELS_FOLDER = 'models'
        if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)

        # Save model using pickle
        model_filename = f"{model_type}_{job_id}.pkl"
        model_path = os.path.join(MODELS_FOLDER, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['message'] = 'Training completed successfully'
        training_jobs[job_id]['progress'] = 100
        training_jobs[job_id]['results'] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_path': model_path
        }

    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['progress'] = 0


def train_selected_model(model_type, X_train, y_train, X_test, y_test, is_classification):
    # Initialize variables
    model = None
    metrics = {}
    feature_importance = {}
    
    # Select and train model
    if model_type == 'linear_regression':
        if is_classification:
            # For classification, use logistic regression
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = LinearRegression()
    
    elif model_type == 'random_forest':
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    elif model_type == 'svm':
        if is_classification:
            model = SVC(probability=True, random_state=42)
        else:
            model = SVR()
    
    elif model_type == 'decision_tree':
        if is_classification:
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
    
    elif model_type == 'knn':
        if is_classification:
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = KNeighborsRegressor(n_neighbors=5)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        metrics['r2_score'] = r2_score(y_test, y_pred)
        metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred)
        metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
        else:
            # For multi-class models, take the mean of absolute coefficients
            feature_importance = dict(zip(X_train.columns, np.mean(np.abs(model.coef_), axis=0)))
    
    return model, metrics, feature_importance

if __name__ == '__main__':
    app.run(debug=True, port=8080)