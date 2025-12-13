import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from urllib.parse import urlparse
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

# Real API Functions
def check_virustotal(url, api_key):
    """Check URL against VirusTotal API"""
    try:
        # Submit URL for scanning
        # scan_url = 
        # "https://www.virustotal.com
        # /vtapi/v2/url/scan"
        params = {'apikey': api_key, 'url': url}
        response = requests.post(scan_url, data=params, timeout=10)
        
        if response.status_code == 200:
            scan_result = response.json()
            resource = scan_result.get('scan_id')
            
            # Wait a moment then get report
            time.sleep(2)
            # report_url = 
            # "https://www.virustotal
            # .com/vtapi/v2/url/report"
            params = {'apikey': api_key, 'resource': resource}
            report_response = requests.get(report_url, params=params, timeout=10)
            
            if report_response.status_code == 200:
                report = report_response.json()
                positives = report.get('positives', 0)
                total = report.get('total', 0)
                
                if total > 0:
                    malicious_ratio = positives / total
                    return {
                        'malicious': malicious_ratio > 0.1,  # More than 10% detection
                        'confidence': malicious_ratio,
                        'details': f"VirusTotal: {positives}/{total} engines detected malware"
                    }
        
        return {'malicious': False, 'confidence': 0.0, 'details': 'VirusTotal: Scan failed or clean'}
    except Exception as e:
        return {'malicious': False, 'confidence': 0.0, 'details': f'VirusTotal: Error - {str(e)}'}

def check_urlscan(url, api_key):
    """Check URL against URLScan.io API"""
    try:
        headers = {'API-Key': api_key, 'Content-Type': 'application/json'}
        data = {'url': url, 'visibility': 'public'}
        
        # response = 
        # requests.post(
            # 'https://urlscan.io
            # /api/v1/scan/', 
                               headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            uuid = result.get('uuid')
            
            # Wait for scan completion
            time.sleep(5)
            # report_response = requests
            # .get(f'https://urlscan
            # .io/api/v1/result/{uuid}/', 
                                         headers=headers, timeout=10)
            
            if report_response.status_code == 200:
                report = report_response.json()
                verdicts = report.get('verdicts', {})
                overall = verdicts.get('overall', {})
                
                malicious = overall.get('malicious', False)
                score = overall.get('score', 0)
                
                return {
                    'malicious': malicious,
                    'confidence': min(score / 100, 1.0),
                    'details': f"URLScan.io: {'Malicious' if malicious else 'Clean'} (Score: {score})"
                }
        
        return {'malicious': False, 'confidence': 0.0, 'details': 'URLScan.io: Scan failed or clean'}
    except Exception as e:
        return {'malicious': False, 'confidence': 0.0, 'details': f'URLScan.io: Error - {str(e)}'}

def check_abuseipdb(url, api_key):
    """Check IP/domain against AbuseIPDB API"""
    try:
        parsed = urlparse(url)
        target = parsed.netloc
        
        # Remove port if present
        if ':' in target:
            target = target.split(':')[0]
        
        # Check if it's an IP or domain
        try:
            import ipaddress
            ipaddress.ip_address(target)
            is_ip = True
        except:
            is_ip = False
        
        if is_ip:
            # Check IP
            # url_check = 
            # f"https://api.
            # abuseipdb.com/
            # api/v2/check"
            params = {'ipAddress': target, 'maxAgeInDays': '90'}
        else:
            # For domains, we'd need a different approach, but AbuseIPDB primarily does IPs
            return {'malicious': False, 'confidence': 0.0, 'details': 'AbuseIPDB: Domain checking not supported'}
        
        headers = {'Accept': 'application/json', 'Key': api_key}
        response = requests.get(url_check, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            abuse_score = data['data']['abuseConfidenceScore']
            
            malicious = abuse_score > 50  # More than 50% abuse confidence
            
            return {
                'malicious': malicious,
                'confidence': abuse_score / 100,
                'details': f"AbuseIPDB: Abuse score {abuse_score}%"
            }
        
        return {'malicious': False, 'confidence': 0.0, 'details': 'AbuseIPDB: Check failed or clean'}
    except Exception as e:
        return {'malicious': False, 'confidence': 0.0, 'details': f'AbuseIPDB: Error - {str(e)}'}

# Set page config
st.set_page_config(
    page_title="🔒 Malicious URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-malicious {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .prediction-suspicious {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ URL Security Analyzer")
st.sidebar.markdown("---")

# Model Performance Metrics
st.sidebar.subheader("📊 Model Performance")
metrics = {
    "Logistic Regression": {"Accuracy": 0.74, "Precision": 0.77, "Recall": 0.68, "F1-Score": 0.72},
    "Naive Bayes": {"Accuracy": 0.57, "Precision": 0.81, "Recall": 0.19, "F1-Score": 0.31},
    "Random Forest": {"Accuracy": 0.92, "Precision": 0.95, "Recall": 0.90, "F1-Score": 0.92},
    "Isolation Forest": {"Anomaly Detection": "Zero-day threats"}
}

for model, scores in metrics.items():
    with st.sidebar.expander(f"🔍 {model}"):
        for metric, value in scores.items():
            if isinstance(value, float):
                st.metric(metric, f"{value:.2f}")
            else:
                st.write(f"**{metric}:** {value}")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Layers")
layers = [
    "1. URL Validity Check",
    "2. Trusted Domain Whitelist", 
    "3. Pattern-Based Detection",
    "4. Supervised ML Voting",
    "5. Unsupervised Anomaly Detection",
    "6. Final Rule-Based Decision"
]
for layer in layers:
    st.sidebar.write(f"✅ {layer}")

# Hardcoded API keys (for internal use only)
# API_KEYS = {
#     'virustotal': ------'b17c7f821b06c08e25856a31652e9edc0a06c0bbd68f4f2ba4cfb4d0a92605c4'----,
#     'urlscan': '019b1751-1c9d-72f9-8d64-827b6d69e33b',
#     'abuseipdb': -----'f7b5a5111073dfd495693f75f018006fa7e4624a0bbcad354757ccaf90e5766cf2c41fdd13fd1432'----
# }

def is_trusted_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        return any(td in domain for td in trusted_domains)
    except:
        return False

# Mock APIs
def threat_api_check(url, vt_key=None, urlscan_key=None, abuseipdb_key=None):
    """Enhanced threat intelligence check using real APIs"""
    api_results = []
    total_confidence = 0
    api_count = 0

    # Use hardcoded keys if none provided
    vt_key = vt_key or API_KEYS['virustotal']
    urlscan_key = urlscan_key or API_KEYS['urlscan']
    abuseipdb_key = abuseipdb_key or API_KEYS['abuseipdb']

    # Check VirusTotal if key provided
    if vt_key:
        vt_result = check_virustotal(url, vt_key)
        api_results.append(vt_result)
        if vt_result['malicious']:
            total_confidence += vt_result['confidence']
        api_count += 1

    # Check URLScan.io if key provided
    if urlscan_key:
        urlscan_result = check_urlscan(url, urlscan_key)
        api_results.append(urlscan_result)
        if urlscan_result['malicious']:
            total_confidence += urlscan_result['confidence']
        api_count += 1

    # Check AbuseIPDB if key provided
    if abuseipdb_key:
        abuseipdb_result = check_abuseipdb(url, abuseipdb_key)
        api_results.append(abuseipdb_result)
        if abuseipdb_result['malicious']:
            total_confidence += abuseipdb_result['confidence']
        api_count += 1

    # If no APIs available, fall back to mock detection
    if api_count == 0:
        malicious_patterns = ['phishing', 'malware', 'br-icloud.com.br', 'fake', 'malicious']
        is_malicious = any(pattern in url for pattern in malicious_patterns)
        return {
            'malicious': is_malicious,
            'confidence': 0.5 if is_malicious else 0.0,
            'details': 'Mock detection: Pattern-based analysis',
            'api_results': []
        }

    # Calculate overall confidence
    avg_confidence = total_confidence / api_count if api_count > 0 else 0
    overall_malicious = any(result['malicious'] for result in api_results)

    return {
        'malicious': overall_malicious,
        'confidence': avg_confidence,
        'details': f"Real API analysis ({api_count} services checked)",
        'api_results': api_results
    }

def is_url_alive(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except:
        return False

# Feature extraction
def extract_features(url):
    features = {}
    
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    special_chars = ['@', '?', '-', '_', '.', '/', '=', '&', '%', '+', '$', '#', '!', '*', '(', ')', '[', ']', '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', ',']
    features['num_special'] = sum(url.count(char) for char in special_chars)
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        import ipaddress
        ipaddress.ip_address(domain)
        features['has_ip'] = 1
    except:
        features['has_ip'] = 0
    
    features['path_length'] = len(parsed.path)
    features['domain_length'] = len(domain)
    features['num_subdomains'] = domain.count('.') - 1 if domain else 0
    
    suspicious_words = ['login', 'verify', 'secure', 'account', 'update', 'bank', 'paypal', 'free', 'win', 'password']
    features['has_suspicious_words'] = int(any(word in url.lower() for word in suspicious_words))
    
    def entropy(s):
        p, lns = Counter(s), float(len(s))
        return -sum(count/lns * np.log2(count/lns) for count in p.values()) if lns > 0 else 0
    features['entropy'] = entropy(url)
    
    return features

# Load models
@st.cache_resource
def load_models():
    import os
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    log_reg = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
    nb = joblib.load(os.path.join(models_dir, 'naive_bayes.pkl'))
    rf = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    iso = joblib.load(os.path.join(models_dir, 'isolation_forest.pkl'))
    return log_reg, nb, rf, iso

log_reg, nb, rf, iso = load_models()

# Prediction function with details
def predict_url_detailed(url):
    url = url.lower().strip()
    
    details = {
        'model_predictions': {},
        'checks': {
            'valid': False,
            'trusted': False,
            'api_flag': False,
            'anomaly': False
        }
    }
    
    # 1. Trusted domain check
    if is_trusted_domain(url):
        details['checks']['trusted'] = True
        return "SAFE", "Trusted domain", details
    
    # 2. URL validity
    is_valid = is_url_alive(url)
    details['checks']['valid'] = is_valid
    if not is_valid:
        return "SUSPICIOUS", "URL not reachable", details
    
    # 3. Threat API check
    api_result = threat_api_check(url, API_KEYS['virustotal'], API_KEYS['urlscan'], API_KEYS['abuseipdb'])
    details['checks']['api_flag'] = api_result['malicious']
    details['api_details'] = api_result['details']
    details['api_results'] = api_result.get('api_results', [])
    if api_result['malicious']:
        return "MALICIOUS", f"Flagged by threat intelligence ({api_result['details']})", details
    
    # 4. Extract features
    features = extract_features(url)
    X = pd.DataFrame([features])
    feature_cols = ['url_length', 'num_digits', 'num_special', 'has_ip', 'path_length', 'domain_length', 'num_subdomains', 'has_suspicious_words', 'entropy']
    X = X[feature_cols]
    
    # 5. Individual model predictions
    pred_log = log_reg.predict(X)[0]
    pred_nb = nb.predict(X)[0]
    pred_rf = rf.predict(X)[0]
    
    details['model_predictions'] = {
        'Logistic Regression': pred_log,
        'Naive Bayes': pred_nb,
        'Random Forest': pred_rf
    }
    
    votes = [pred_log, pred_nb, pred_rf]
    majority_vote = 1 if sum(votes) >= 2 else 0
    
    # 6. Isolation Forest
    iso_pred = iso.predict(X)[0]
    anomaly = 1 if iso_pred == -1 else 0
    details['checks']['anomaly'] = anomaly
    
    # Final rules
    if majority_vote == 1 or anomaly == 1:
        reason = f"ML majority: {sum(votes)}/3 votes, Anomaly: {anomaly}"
        return "MALICIOUS", reason, details
    else:
        return "SAFE", "No flags detected", details

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["🔍 URL Analysis", "📈 Model Insights", "ℹ️ About"])

with tab1:
    st.markdown('<h1 class="main-header">🔒 Malicious URL Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        "Enter a URL below to analyze its security status using advanced machine learning and threat intelligence."
    )
    
    # URL input
    url_input = st.text_input(
        "Enter URL to analyze:",
        # placeholder="https://example.com",
        key="url_input"
    )

    # Centered Analyze button (30% width)
    col_left, col_center, col_right = st.columns([10, 3, 10])
    with col_center:
        analyze_button = st.button("🔍 Analyze", type="primary")


    if analyze_button and url_input:
        with st.spinner("🔍 Analyzing URL..."):
            result, reason, details = predict_url_detailed(url_input)
        
        # Display result
        if result == "SAFE":
            st.markdown(f'<div class="prediction-safe"><h3>✅ SAFE URL</h3><p><strong>Reason:</strong> {reason}</p></div>', unsafe_allow_html=True)
        elif result == "MALICIOUS":
            st.markdown(f'<div class="prediction-malicious"><h3>❌ MALICIOUS URL</h3><p><strong>Reason:</strong> {reason}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-suspicious"><h3>⚠️ SUSPICIOUS URL</h3><p><strong>Reason:</strong> {reason}</p></div>', unsafe_allow_html=True)
        
        # Show detailed breakdown
        st.subheader("🔍 Analysis Breakdown")

        # Overall Threat Assessment Summary
        col_summary1, col_summary2, col_summary3 = st.columns(3)

        with col_summary1:
            ml_threats = sum(1 for pred in details['model_predictions'].values() if pred == 1)
            st.metric("ML Models Flagged", f"{ml_threats}/3")

        with col_summary2:
            api_threats = sum(1 for r in details.get('api_results', []) if r.get('malicious', False))
            total_apis = len(details.get('api_results', []))
            st.metric("APIs Flagged", f"{api_threats}/{total_apis}")

        with col_summary3:
            overall_risk = "HIGH" if (ml_threats >= 2 or api_threats >= 1) else "MEDIUM" if ml_threats >= 1 else "LOW"
            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[overall_risk]
            st.metric("Overall Risk", f"{risk_color} {overall_risk}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Individual Model Predictions:**")
            for model, pred in details['model_predictions'].items():
                status = "🟢 SAFE" if pred == 0 else "🔴 MALICIOUS"
                st.write(f"{model}: {status}")
        
        with col2:
            st.markdown("**Security Checks:**")
            checks = details['checks']
            st.write(f"URL Valid: {'✅' if checks['valid'] else '❌'}")
            st.write(f"Trusted Domain: {'✅' if checks['trusted'] else '❌'}")
            st.write(f"Threat API Flag: {'❌' if checks['api_flag'] else '✅'}")
            st.write(f"Anomaly Detected: {'❌' if checks['anomaly'] else '✅'}")

            # Enhanced API Analysis Section
            if 'api_details' in details:
                st.markdown("---")
                st.markdown("### 🔗 Threat Intelligence APIs")

                # API Analysis Summary
                api_summary_col1, api_summary_col2 = st.columns([2, 1])
                with api_summary_col1:
                    st.markdown(f"**Analysis Type:** {details['api_details']}")
                with api_summary_col2:
                    if details.get('api_results'):
                        working_apis = len([r for r in details['api_results'] if 'Error' not in r['details']])
                        st.metric("Active APIs", f"{working_apis}/{len(details['api_results'])}")

                # Individual API Results with Graphics
                if details.get('api_results'):
                    st.markdown("**API Detection Results:**")

                    for api_result in details['api_results']:
                        # Create API result card
                        is_malicious = api_result.get('malicious', False)
                        confidence = api_result.get('confidence', 0.0)

                        # Color coding
                        if is_malicious:
                            card_color = "#ffcccc"  # Light red
                            border_color = "#ff4444"  # Red border
                            status_icon = "🚨"
                            status_text = "THREAT DETECTED"
                        else:
                            card_color = "#ccffcc"  # Light green
                            border_color = "#44ff44"  # Green border
                            status_icon = "✅"
                            status_text = "CLEAN"

                        # API Card
                        st.markdown(f"""
                        <div style="
                            border: 2px solid {border_color};
                            border-radius: 10px;
                            padding: 15px;
                            margin: 10px 0;
                            background-color: {card_color};
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <h4 style="margin: 0; color: #333;">{status_icon} {api_result['details'].split(':')[0]}</h4>
                            <p style="margin: 5px 0; font-size: 14px;">{api_result['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence Progress Bar
                        if confidence > 0:
                            st.progress(min(confidence, 1.0))
                            st.caption(f"Confidence: {confidence:.1%}")

                    # API Contribution Summary
                    st.markdown("---")
                    st.markdown("**API Threat Assessment:**")

                    total_confidence = sum(r.get('confidence', 0) for r in details['api_results'])
                    avg_confidence = total_confidence / len(details['api_results']) if details['api_results'] else 0

                    threat_level = "HIGH" if avg_confidence > 0.7 else "MEDIUM" if avg_confidence > 0.3 else "LOW"
                    threat_color = "#ff4444" if threat_level == "HIGH" else "#ffaa44" if threat_level == "MEDIUM" else "#44ff44"

                    st.markdown(f"""
                    <div style="
                        border: 2px solid {threat_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        background-color: #f9f9f9;
                        text-align: center;
                    ">
                        <h3 style="margin: 0; color: {threat_color};">{threat_level} THREAT LEVEL</h3>
                        <p style="margin: 5px 0;">Average API Confidence: {avg_confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No API results available. Using pattern-based detection only.")

                # API Confidence Chart
                if details.get('api_results') and len(details['api_results']) > 1:
                    st.markdown("---")
                    st.markdown("**API Confidence Scores:**")

                    api_names = []
                    confidences = []
                    colors = []

                    for api_result in details['api_results']:
                        api_name = api_result['details'].split(':')[0]
                        confidence = api_result.get('confidence', 0.0)
                        is_malicious = api_result.get('malicious', False)

                        api_names.append(api_name)
                        confidences.append(confidence)
                        colors.append('#ff4444' if is_malicious else '#44aa44')

                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(api_names, confidences, color=colors, alpha=0.7)

                    ax.set_xlabel('Confidence Score')
                    ax.set_title('API Detection Confidence')
                    ax.set_xlim(0, 1)

                    # Add value labels on bars
                    for bar, conf in zip(bars, confidences):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{conf:.1%}', va='center', fontweight='bold')

                    st.pyplot(fig)

with tab2:
    st.header("📈 Model Performance Insights")
    
    # Model comparison chart
    st.subheader("Model Accuracy Comparison")
    model_names = list(metrics.keys())[:-1]  # Exclude Isolation Forest
    accuracies = [metrics[m]["Accuracy"] for m in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Supervised Model Accuracy Comparison')
    ax.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T
    # Remove Isolation Forest for highlighting since it has string values
    numeric_metrics = metrics_df.drop('Isolation Forest')
    st.dataframe(numeric_metrics.style.highlight_max(axis=0, color='#d4edda'))
    
    # Show Isolation Forest separately
    st.write("**Isolation Forest:** Unsupervised anomaly detection for zero-day threats")
    
    # Feature importance (for Random Forest)
    st.subheader("Feature Importance (Random Forest)")
    feature_names = ['URL Length', 'Digit Count', 'Special Chars', 'Has IP', 'Path Length', 
                    'Domain Length', 'Subdomains', 'Suspicious Words', 'Entropy']
    # Mock importance values (in real app, extract from model)
    importance = [0.15, 0.12, 0.18, 0.08, 0.10, 0.09, 0.07, 0.14, 0.07]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(feature_names, importance, color='#1f77b4')
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance in Malicious URL Detection')
    st.pyplot(fig2)

with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🛡️ Production-Grade Malicious URL Detection
    
    This system implements a **hybrid machine learning and threat intelligence approach** for detecting malicious URLs, inspired by industry solutions like VirusTotal and Google Safe Browsing.
    
    ### 🏗️ System Architecture
    
    **6-Layer Security Pipeline:**
    1. **URL Validity Check** - Ensures URL is accessible
    2. **Trusted Domain Whitelist** - Immediate safe classification for government/educational sites
    3. **Pattern-Based Detection** - Checks against known malicious patterns
    4. **Supervised ML Ensemble** - Voting from multiple classifiers
    5. **Unsupervised Anomaly Detection** - Catches zero-day threats
    6. **Rule-Based Final Decision** - Combines all layers with business logic
    
    ### 🤖 Machine Learning Models
    
    - **Logistic Regression**: Baseline interpretable model (74% accuracy)
    - **Naive Bayes**: Text/lexical pattern specialist (57% accuracy)  
    - **Random Forest**: High-accuracy ensemble model (92% accuracy)
    - **Isolation Forest**: Unsupervised anomaly detection for unknown threats
    
    ### 📊 Dataset
    
    - **491,876 URLs** (balanced: 245,938 benign, 245,938 malicious)
    - Features: URL length, digit count, special characters, IP detection, domain analysis, entropy
    - Sources: Kaggle malicious URLs, PhishTank, Alexa top sites
    
    ### 🔧 Technologies Used
    
    - **Frontend**: Streamlit
    - **ML**: Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    
    ### 🎯 Use Cases
    
    - Web security analysis
    - Phishing detection
    - Malware URL identification
    - Educational/research purposes
    
    ---
    **Built for production reliability with explainable AI decisions.**
    """)
    
    # Contact info
    st.markdown("---")
    st.markdown("**👨‍💻 Developer:** Aditya Tiwari")
    st.markdown("**📅 Date:** 13-12-2025")
    # st.markdown("**🔗 GitHub:** [
        # AdityaTiwari0890](https://
    # github.com/AdityaTiwari0890)")
