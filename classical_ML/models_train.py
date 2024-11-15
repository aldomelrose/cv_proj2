import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score
from preprocessing import prepare_data
from sklearn.preprocessing import LabelEncoder
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io


def load_processed_data(timestamp):
    """Load processed data from saved files."""
    processed_dir = 'processed_data'
    # Replace with npy files
    try:
        X_train = np.load(os.path.join(processed_dir, f'X_test_20241112_213136.npy'))
        X_test = np.load(os.path.join(processed_dir, f'X_test_20241112_213136.npy'))
        y_train = np.load(os.path.join(processed_dir, f'y_test_20241112_213136.npy'))
        y_test = np.load(os.path.join(processed_dir, f'y_test_20241112_213136.npy'))
        stats = joblib.load(os.path.join(processed_dir, f'processing_stats_{timestamp}.joblib'))
        
        print("Loaded processed data:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, stats
        
    except Exception as e:
        raise Exception(f"Error loading processed data: {str(e)}")


def create_results_dir():
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/trial1_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['per_class'] = {
        label: {
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1-score': scores['f1-score'],
            'support': scores['support']
        }
        for label, scores in class_report.items()
        if label not in ['accuracy', 'macro avg', 'weighted avg']
    }
    
    # Multi-class ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    except:
        metrics['roc_auc'] = None  # In case of issues with probabilities
        
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save detailed confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    return cm

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, save_dir):
    """Train, evaluate, and save the model with comprehensive metrics."""
    # Train
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Convert predictions back to original labels if they're numeric
    if isinstance(y_test[0], (int, np.integer)):
        label_encoder = LabelEncoder()
        label_encoder.fit(sorted(list(set(y_train))))  # Fit on unique sorted values
        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)
    
    
    # Get evaluation metrics
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(save_dir, f'{model_name.lower()}_confusion_matrix.png')
    )
    
    # Save model
    model_path = os.path.join(save_dir, f'{model_name.lower()}_model.joblib')
    joblib.dump(model, model_path)
    
    # Save info file with necessary metadata
    info = {
        'label_mapping': {label: label for label in np.unique(y_train)},
        'feature_shape': X_train.shape[1],
        'num_classes': len(np.unique(y_train))
    }
    info_path = os.path.join(save_dir, f'{model_name.lower()}_info.joblib')
    joblib.dump(info, info_path)
    
    # Save detailed results separately
    results = {
        'metrics': metrics,
        'cross_val_scores': {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
    }
    results_path = os.path.join(save_dir, f'{model_name.lower()}_results.joblib')
    joblib.dump(results, results_path)
    
    # Print detailed evaluation
    print(f"\n{model_name} Evaluation Results:")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    if metrics['roc_auc']:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"\nCross-validation scores:")
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nPer-class Performance:")
    for label, scores in metrics['per_class'].items():
        print(f"\nClass {label}:")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall: {scores['recall']:.4f}")
        print(f"F1-score: {scores['f1-score']:.4f}")
        print(f"Support: {scores['support']}")
    
    return model, results

def save_training_summary(results_dir, X_train, X_test, y_train, y_test, model_results):
    """Save comprehensive training summary."""
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write("Sign Language Recognition - Trial 1 Results\n")
        f.write("=======================================\n\n")
        
        # Dataset information
        f.write("Dataset Information:\n")
        f.write("-----------------\n")
        f.write(f"Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(X_train) + len(X_test)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(y_train))}\n\n")
        
        # Class distribution
        f.write("Class Distribution:\n")
        f.write("-----------------\n")
        for label in sorted(np.unique(y_train)):
            train_count = np.sum(y_train == label)
            test_count = np.sum(y_test == label)
            f.write(f"Class {label}:\n")
            f.write(f"  Training: {train_count}\n")
            f.write(f"  Testing: {test_count}\n")
            f.write(f"  Total: {train_count + test_count}\n\n")
        
        # Model performance comparison
        f.write("\nModel Performance Comparison:\n")
        f.write("---------------------------\n")
        for model_name, results in model_results.items():
            f.write(f"\n{model_name}:\n")
            metrics = results['metrics']
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
            if metrics['roc_auc']:
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"Cross-validation mean: {results['cross_val_scores']['mean']:.4f}\n")
            f.write(f"Cross-validation std: {results['cross_val_scores']['std']:.4f}\n")

def create_training_report(results_dir, model_results, X_train, X_test, y_train, y_test):
    """Create a comprehensive PDF report of training results."""
    pdf_path = os.path.join(results_dir, 'training_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Sign Language Recognition - Training Report", title_style))
    elements.append(Spacer(1, 12))

    # Dataset Information
    elements.append(Paragraph("Dataset Information", styles['Heading2']))
    dataset_info = [
        ["Total Samples", str(len(X_train) + len(X_test))],
        ["Training Samples", str(len(X_train))],
        ["Testing Samples", str(len(X_test))],
        ["Number of Features", str(X_train.shape[1])],
        ["Number of Classes", str(len(np.unique(y_train)))]
    ]
    t = Table(dataset_info)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Class Distribution Visualization
    elements.append(Paragraph("Class Distribution", styles['Heading2']))
    plt.figure(figsize=(10, 6))
    class_counts = np.array([np.sum(y_train == label) for label in sorted(np.unique(y_train))])
    plt.bar(sorted(np.unique(y_train)), class_counts)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Add plot to PDF
    img = Image(buf)
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    elements.append(img)
    elements.append(Spacer(1, 20))

    # Model Results
    elements.append(Paragraph("Model Performance Comparison", styles['Heading2']))
    
    for model_name, results in model_results.items():
        elements.append(Paragraph(f"{model_name} Results:", styles['Heading3']))
        metrics = results['metrics']
        
        # Create table of metrics
        metrics_data = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["Macro Precision", f"{metrics['precision_macro']:.4f}"],
            ["Macro Recall", f"{metrics['recall_macro']:.4f}"],
            ["Macro F1", f"{metrics['f1_macro']:.4f}"],
            ["ROC AUC", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"],
            ["CV Mean", f"{results['cross_val_scores']['mean']:.4f}"],
            ["CV Std", f"{results['cross_val_scores']['std']:.4f}"]
        ]
        
        t = Table(metrics_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

        # Add confusion matrix
        elements.append(Paragraph("Confusion Matrix:", styles['Heading4']))
        cm_path = os.path.join(results_dir, f'{model_name.lower()}_confusion_matrix.png')
        if os.path.exists(cm_path):
            img = Image(cm_path)
            img.drawHeight = 6*inch
            img.drawWidth = 6*inch
            elements.append(img)
        elements.append(Spacer(1, 20))

    # Build PDF
    doc.build(elements)
    print(f"Training report saved to: {pdf_path}")

def main():
    # Configuration
    DATA_DIR = 'hands_dataset_cleaned'
    USE_PROCESSED_DATA = True
    PROCESSED_TIMESTAMP = "20241112_213136"
    
    try:
        # Get data (either load processed or process fresh)
        if USE_PROCESSED_DATA:
            print(f"Loading processed data from timestamp: {PROCESSED_TIMESTAMP}")
            X_train, X_test, y_train, y_test, _ = load_processed_data(PROCESSED_TIMESTAMP)
        else:
            print("Processing fresh data...")
            X_train, X_test, y_train, y_test = prepare_data(DATA_DIR)
        
        # Create label encoder and transform labels to numbers
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Create results directory
        results_dir = create_results_dir()
        print(f"Saving results to: {results_dir}")
        
        model_results = {}
        
        # Train and evaluate Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,           # Limit depth
            min_samples_split=5,    # Require more samples to split
            min_samples_leaf=2,     # Require more samples in leaves
            max_features='sqrt',    # Use sqrt of features
            random_state=42
        )
        _, rf_results = train_and_evaluate(
            rf_model, X_train, X_test, y_train, y_test,  # Use original labels
            "RandomForest", results_dir
        )
        model_results['RandomForest'] = rf_results
        
        # Train and evaluate SVM
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        _, svm_results = train_and_evaluate(
            svm_model, X_train, X_test, y_train, y_test,  # Use original labels
            "SVM", results_dir
        )
        model_results['SVM'] = svm_results
        
        # Train and evaluate XGBoost (use encoded labels)
        xgb_model = XGBClassifier(
            n_estimators=100,      
            max_depth=3,           
            learning_rate=0.1,     
            min_child_weight=1,    
            subsample=0.8,         
            colsample_bytree=0.8,  
            random_state=42
        )
        
        # Modify train_and_evaluate function call for XGBoost
        _, xgb_results = train_and_evaluate(
            xgb_model, 
            X_train, X_test, 
            y_train_encoded, y_test_encoded,  # Use encoded labels
            "XGBoost", results_dir
        )
        model_results['XGBoost'] = xgb_results
        
        # Save comprehensive summary
        save_training_summary(results_dir, X_train, X_test, y_train, y_test, model_results)
        
        # Generate PDF report
        create_training_report(results_dir, model_results, X_train, X_test, y_train, y_test)

        print(f"\nTraining complete! Results saved to: {results_dir}")
        


    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()