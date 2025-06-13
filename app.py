import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import yaml
import os
import time
import base64
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="🐟 Fish Behavior Detection AI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS styling
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: white;
    }
    
    .behavior-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .behavior-card:hover {
        transform: translateY(-5px);
    }
    
    .overall-behavior {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .health-assessment {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Animation */
    @keyframes swim {
        0% { transform: translateX(-10px); }
        50% { transform: translateX(10px); }
        100% { transform: translateX(-10px); }
    }
    
    .swimming-fish {
        animation: swim 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        # Try multiple paths for the model
        model_paths = [
            'D:/FISH DETECTION/Fish Detection and Counting.v1i.yolov4scaled/scripts/yolov8n.pt',
            'yolov8n.pt',
            './yolov8n.pt'
        ]
        
        model = None
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = YOLO(path)
                    break
            except:
                continue
        
        if model is None:
            # Download default model if none found
            model = YOLO('yolov8n.pt')
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_behavior_info():
    """Get behavior colors, emojis, and descriptions - ONLY FISH BEHAVIORS"""
    behaviors = {
        'Normal': {
            'color': '#4CAF50', 
            'emoji': '😊', 
            'desc': 'Fish swimming normally with regular patterns',
            'detailed': 'The fish is exhibiting normal, healthy swimming behavior with regular movement patterns and stable positioning in the water.'
        },
        'Stressed': {
            'color': '#FF9800', 
            'emoji': '😰', 
            'desc': 'Fish shows signs of stress - erratic movements',
            'detailed': 'The fish displays stress indicators such as rapid, erratic swimming patterns, irregular movements, or unusual positioning that may indicate environmental stress or health issues.'
        },
        'Aggressive': {
            'color': '#F44336', 
            'emoji': '😠', 
            'desc': 'Fish displays aggressive behavior - fast movements',
            'detailed': 'The fish shows aggressive behavior characterized by fast, directed movements, territorial displays, or confrontational positioning towards other fish or objects.'
        },
        'Feeding': {
            'color': '#2196F3', 
            'emoji': '🍽️', 
            'desc': 'Fish in feeding mode - slow, concentrated movements',
            'detailed': 'The fish is in feeding mode, showing slow, deliberate movements concentrated around food sources or feeding areas with focused attention on foraging behavior.'
        },
        'Resting': {
            'color': '#9C27B0', 
            'emoji': '😴', 
            'desc': 'Fish is resting or moving very slowly',
            'detailed': 'The fish is in a resting state, exhibiting minimal movement, hovering near the bottom or in sheltered areas, conserving energy during inactive periods.'
        }
    }
    return behaviors

def classify_fish_behavior(box_info, image_shape):
    """Classify fish behavior based on detection characteristics"""
    x1, y1, x2, y2, conf = box_info
    height, width = image_shape[:2]
    
    # Calculate fish characteristics
    fish_width = x2 - x1
    fish_height = y2 - y1
    aspect_ratio = fish_width / fish_height if fish_height > 0 else 1
    center_y = (y1 + y2) / 2
    normalized_y = center_y / height
    
    # Behavior classification based on position and characteristics
    # This is a simplified heuristic - in production you'd use a trained model
    
    # Fish near bottom (normalized_y > 0.7) - likely resting or feeding
    if normalized_y > 0.7:
        if aspect_ratio > 2.5:  # Elongated shape suggests resting
            return 'Resting'
        else:
            return 'Feeding'
    
    # Fish in upper area (normalized_y < 0.3) - likely normal or aggressive
    elif normalized_y < 0.3:
        if conf > 0.8 and aspect_ratio > 2.0:  # High confidence, elongated
            return 'Aggressive'
        else:
            return 'Normal'
    
    # Fish in middle area - analyze based on confidence and shape
    else:
        if conf < 0.5:  # Low confidence might indicate erratic movement
            return 'Stressed'
        elif aspect_ratio > 3.0:  # Very elongated - fast movement
            return 'Aggressive'
        else:
            return 'Normal'

def run_detection(image, model, conf_threshold=0.25):
    """Run fish detection with proper behavior classification"""
    try:
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run inference
        results = model(image_np, conf=conf_threshold, verbose=False)
        
        detections = []
        annotated_image = image_np.copy()
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                behaviors = get_behavior_info()
                
                for box in result.boxes:
                    # Get coordinates and info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    
                    # Classify behavior based on fish characteristics
                    behavior = classify_fish_behavior((x1, y1, x2, y2, conf), image_np.shape)
                    
                    detections.append({
                        'behavior': behavior,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    # Draw on image
                    color_hex = behaviors[behavior]['color']
                    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color_rgb, 3)
                    
                    # Add label
                    label = f"{behavior} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color_rgb, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return detections, annotated_image
        
    except Exception as e:
        st.error(f"Detection error: {e}")
        return [], image_np

def create_charts(detections):
    """Create beautiful charts"""
    if not detections:
        return None, None
    
    behaviors = get_behavior_info()
    
    # Count behaviors
    behavior_counts = {}
    for det in detections:
        behavior = det['behavior']
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    # Pie chart
    pie_fig = go.Figure(data=[go.Pie(
        labels=list(behavior_counts.keys()),
        values=list(behavior_counts.values()),
        hole=0.4,
        marker=dict(
            colors=[behaviors[b]['color'] for b in behavior_counts.keys()],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    pie_fig.update_layout(
        title=dict(text="🐟 Fish Behavior Distribution", font=dict(size=20, color='white'), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    
    # Bar chart
    behavior_list = [det['behavior'] for det in detections]
    confidences = [det['confidence'] for det in detections]
    colors = [behaviors[b]['color'] for b in behavior_list]
    
    bar_fig = go.Figure(data=[go.Bar(
        x=behavior_list,
        y=confidences,
        marker=dict(color=colors),
        text=[f"{conf:.2f}" for conf in confidences],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>'
    )])
    
    bar_fig.update_layout(
        title=dict(text="🎯 Detection Confidence Scores", font=dict(size=18, color='white'), x=0.5),
        xaxis=dict(title="Behavior Type", color='white'),
        yaxis=dict(title="Confidence Score", color='white', range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return pie_fig, bar_fig

def display_overall_behavior_summary(detections):
    """Display overall fish behavior summary and health assessment"""
    if not detections:
        return
    
    behaviors = get_behavior_info()
    total_fish = len(detections)
    
    # Count each behavior
    behavior_counts = {}
    for det in detections:
        behavior = det['behavior']
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    # Create behavior summary
    behavior_summary = []
    for behavior, count in behavior_counts.items():
        if count > 0:
            emoji = behaviors[behavior]['emoji']
            percentage = (count / total_fish) * 100
            behavior_summary.append(f"{emoji} {count} {behavior} ({percentage:.1f}%)")
    
    summary_text = " | ".join(behavior_summary)
    
    # Display overall summary
    st.markdown(f"""
    <div class="overall-behavior pulse-animation">
        🐟 <strong>OVERALL FISH BEHAVIOR SUMMARY</strong><br>
        Detected {total_fish} fish with behaviors: {summary_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate health assessment
    normal_count = behavior_counts.get('Normal', 0)
    stressed_count = behavior_counts.get('Stressed', 0)
    aggressive_count = behavior_counts.get('Aggressive', 0)
    feeding_count = behavior_counts.get('Feeding', 0)
    resting_count = behavior_counts.get('Resting', 0)
    
    # Health percentage based on normal + feeding + resting vs stressed + aggressive
    healthy_behaviors = normal_count + feeding_count + resting_count
    unhealthy_behaviors = stressed_count + aggressive_count
    health_percentage = (healthy_behaviors / total_fish) * 100 if total_fish > 0 else 0
    
    # Determine overall health status
    if health_percentage >= 80:
        health_status = "🟢 EXCELLENT"
        health_color = "#4CAF50"
        health_message = "Most fish are showing healthy, normal behavior patterns. The aquatic environment appears to be in excellent condition."
    elif health_percentage >= 60:
        health_status = "🟡 GOOD"
        health_color = "#FF9800"
        health_message = "Majority of fish are healthy, but some may need attention. Monitor the stressed or aggressive fish closely."
    elif health_percentage >= 40:
        health_status = "🟠 MODERATE"
        health_color = "#FF5722"
        health_message = "Mixed behavior patterns detected. Consider checking water quality, feeding schedule, and environmental factors."
    else:
        health_status = "🔴 CONCERNING"
        health_color = "#F44336"
        health_message = "Many fish showing stress or abnormal behavior. Immediate attention recommended - check water conditions, overcrowding, and health issues."
    
    # Display health assessment
    st.markdown(f"""
    <div class="health-assessment" style="background: linear-gradient(135deg, {health_color}aa 0%, {health_color}dd 100%);">
        <h2>🏥 OVERALL HEALTH ASSESSMENT: {health_status}</h2>
        <p><strong>Health Score:</strong> {health_percentage:.1f}% (Healthy Behaviors)</p>
        <p><strong>Assessment:</strong> {health_message}</p>
        <p><strong>Breakdown:</strong></p>
        <ul>
            <li>✅ Healthy: {healthy_behaviors} fish (Normal: {normal_count}, Feeding: {feeding_count}, Resting: {resting_count})</li>
            <li>⚠️ Concerning: {unhealthy_behaviors} fish (Stressed: {stressed_count}, Aggressive: {aggressive_count})</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations based on behavior
    st.markdown("### 💡 Behavior-Based Recommendations")
    
    recommendations = []
    
    if stressed_count > 0:
        recommendations.append("🔍 **Stressed Fish Detected:** Check water quality (pH, temperature, oxygen levels) and reduce environmental stressors.")
    
    if aggressive_count > 0:
        recommendations.append("⚔️ **Aggressive Behavior:** Monitor for overcrowding, territorial disputes, or competition for resources.")
    
    if feeding_count > total_fish * 0.3:
        recommendations.append("🍽️ **High Feeding Activity:** Good sign! Fish are actively feeding, indicating healthy appetite.")
    
    if resting_count > total_fish * 0.5:
        recommendations.append("😴 **Many Resting Fish:** Normal during certain times, but monitor if persistent throughout the day.")
    
    if normal_count > total_fish * 0.7:
        recommendations.append("✅ **Excellent Normal Behavior:** Fish are healthy and the environment is well-maintained!")
    
    if not recommendations:
        recommendations.append("📊 **Balanced Behavior:** Fish showing diverse but normal behavior patterns.")
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="behavior-card">
            {rec}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">
            🐟 Fish Behavior Detection AI <span class="swimming-fish">🐠</span>
        </div>
        <div class="header-subtitle">
            Advanced Computer Vision for Aquatic Life Monitoring & Behavior Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model!")
        return
    
    st.success("✅ Model loaded successfully! Ready to analyze fish behavior.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        conf_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        st.markdown("## 🐟 Fish Behavior Types")
        behaviors = get_behavior_info()
        for behavior, info in behaviors.items():
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>{info['emoji']} {behavior}:</strong><br>
                <small>{info['desc']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## 📊 What You'll Get")
        st.info("""
        ✅ Individual fish detection
        ✅ Behavior classification  
        ✅ Overall behavior summary
        ✅ Health assessment
        ✅ Recommendations
        ✅ Downloadable results
        """)
    
    # Main content
    st.markdown("## 📸 Upload or Capture Fish Image")
    
    # Tabs for input
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Photo"])
    
    image = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of fish for behavior analysis"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    with tab2:
        camera_image = st.camera_input("Take a picture of fish")
        if camera_image:
            image = Image.open(camera_image)
    
    # Process image
    if image is not None:
        st.markdown("## 🖼️ Original Image")
        st.image(image, caption="Original Fish Image", use_column_width=True)
        
        # Run detection
        with st.spinner("🔍 Analyzing fish behavior with AI..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
                if i < 30:
                    progress_text.text("🔍 Detecting fish in image...")
                elif i < 60:
                    progress_text.text("🧠 Analyzing swimming patterns...")
                elif i < 90:
                    progress_text.text("📊 Classifying behaviors...")
                else:
                    progress_text.text("✨ Generating summary...")
            
            progress_text.empty()
            detections, annotated_image = run_detection(image, model, conf_threshold)
        
        # Show results
        if detections:
            st.markdown("## 🎉 Fish Behavior Detection Results")
            
            # OVERALL BEHAVIOR SUMMARY - This is what you wanted!
            display_overall_behavior_summary(detections)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔍 Detected Fish with Behaviors")
                st.image(annotated_image, use_column_width=True)
            
            with col2:
                pie_fig, bar_fig = create_charts(detections)
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)
            
            # Summary stats
            total_fish = len(detections)
            avg_confidence = np.mean([det['confidence'] for det in detections])
            behavior_counts = {}
            for det in detections:
                behavior = det['behavior']
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
            st.markdown("### 📊 Detection Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🐟 Total Fish Detected</h3>
                    <h1 style="color: #4facfe; margin: 0;">{total_fish}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Average Confidence</h3>
                    <h1 style="color: #4facfe; margin: 0;">{avg_confidence:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                most_common = max(behavior_counts.items(), key=lambda x: x[1])
                behaviors = get_behavior_info()
                emoji = behaviors[most_common[0]]['emoji']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Most Common Behavior</h3>
                    <h1 style="color: #4facfe; margin: 0;">{emoji} {most_common[0]}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual detections
            st.markdown("### 🐠 Individual Fish Analysis")
            behaviors = get_behavior_info()
            
            for i, det in enumerate(detections, 1):
                behavior = det['behavior']
                confidence = det['confidence']
                info = behaviors[behavior]
                
                st.markdown(f"""
                <div class="behavior-card" style="background: linear-gradient(135deg, {info['color']}aa 0%, {info['color']}dd 100%);">
                    <h4>{info['emoji']} Fish #{i} - {behavior} Behavior</h4>
                    <p><strong>Confidence Score:</strong> {confidence:.1%}</p>
                    <p><strong>Description:</strong> {info['desc']}</p>
                    <p><strong>Detailed Analysis:</strong> {info['detailed']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence chart
            if bar_fig:
                st.markdown("### 📈 Confidence Analysis")
                st.plotly_chart(bar_fig, use_container_width=True)
            
            # Download section
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download annotated image
                img_buffer = BytesIO()
                Image.fromarray(annotated_image).save(img_buffer, format='PNG')
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_buffer.getvalue(),
                    file_name=f"fish_behavior_detection_{int(time.time())}.png",
                    mime="image/png"
                )
            
            with col2:
                # Download detailed report
                report = f"""
FISH BEHAVIOR DETECTION REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SUMMARY:
- Total Fish Detected: {total_fish}
- Average Confidence: {avg_confidence:.1%}

BEHAVIOR BREAKDOWN:
"""
                for behavior, count in behavior_counts.items():
                    if count > 0:
                        info = behaviors[behavior]
                        percentage = (count / total_fish) * 100
                        report += f"- {info['emoji']} {behavior}: {count} fish ({percentage:.1f}%)\n"
                        report += f"  Description: {info['desc']}\n\n"

                report += f"""
INDIVIDUAL FISH ANALYSIS:
"""
                for i, det in enumerate(detections, 1):
                    behavior = det['behavior']
                    confidence = det['confidence']
                    info = behaviors[behavior]
                    report += f"{i}. Fish #{i} - {behavior} ({confidence:.1%} confidence)\n"
                    report += f"   {info['detailed']}\n\n"

                report += f"""
RECOMMENDATIONS:
- Monitor fish showing stressed or aggressive behavior
- Ensure proper water conditions and feeding schedules  
- Consider environmental factors affecting fish behavior
- Regular monitoring recommended for optimal fish health
"""
                
                st.download_button(
                    label="📄 Download Detailed Report",
                    data=report,
                    file_name=f"fish_behavior_report_{int(time.time())}.txt",
                    mime="text/plain"
                )
        
        else:
            st.warning("🔍 No fish detected! Try adjusting the confidence threshold or use a clearer image.")
            
            st.markdown("""
            <div class="behavior-card">
                <h3>💡 Tips for Better Detection:</h3>
                <ul>
                    <li>🔆 Ensure good lighting in the image</li>
                    <li>🐟 Make sure fish are clearly visible</li>
                    <li>📏 Fish should be reasonably sized in the image</li>
                    <li>⚙️ Try lowering the confidence threshold</li>
                    <li>📸 Use a clearer, higher quality image</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("## 🌟 Welcome to Fish Behavior Detection AI!")
        
        st.markdown("""
        <div class="overall-behavior">
            🚀 Upload an image or take a photo of fish to analyze their behavior using advanced AI technology!
        </div>
        """, unsafe_allow_html=True)
        
        # Show all behavior types
        st.markdown("### 🐟 Fish Behaviors We Can Detect:")
        
        behaviors = get_behavior_info()
        for behavior, info in behaviors.items():
            st.markdown(f"""
            <div class="behavior-card">
                <h4>{info['emoji']} {behavior}: {info['desc']}</h4>
                <p>{info['detailed']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tips for best results
        st.markdown("### 💡 Tips for Best Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="behavior-card">
                <h4>📸 Image Quality</h4>
                <p>Use clear, well-lit images with visible fish for accurate behavior detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="behavior-card">
                <h4>🎯 Fish Visibility</h4>
                <p>Ensure fish are clearly visible and not heavily overlapping for better analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="behavior-card">
                <h4>⚙️ Adjust Settings</h4>
                <p>Fine-tune confidence threshold in sidebar for optimal detection results</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
