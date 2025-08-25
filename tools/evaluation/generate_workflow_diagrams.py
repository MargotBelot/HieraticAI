#!/usr/bin/env python3
"""
This script creates visualizations for the HieraticAI project workflow,
system architecture, and validation process diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow, ConnectionPatch
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for professional plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_system_architecture_diagram():
    """Create a detailed system architecture visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    fig.suptitle('HieraticAI System Architecture', fontsize=24, fontweight='bold', y=0.95)
    
    # Define colors
    colors = {
        'input': '#3498db',      # Blue
        'ai': '#e74c3c',         # Red  
        'validation': '#27ae60', # Green
        'database': '#f39c12',   # Orange
        'output': '#9b59b6'      # Purple
    }
    
    # Layer 1: Input Layer
    input_box = FancyBboxPatch((0.5, 11), 4, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 12, 'INPUT LAYER', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(2.5, 11.5, '• Westcar Papyrus Images\n• High-resolution scans\n• TIFF/PNG format', 
            fontsize=10, ha='center', va='center')
    
    # Layer 2: AI Processing
    ai_box = FancyBboxPatch((6, 11), 8, 2, boxstyle="round,pad=0.1", 
                            facecolor=colors['ai'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(ai_box)
    ax.text(10, 12.3, 'AI DETECTION PIPELINE', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # AI sub-components
    resnet_box = Rectangle((6.5, 11.6), 2.2, 0.6, facecolor='white', alpha=0.9, edgecolor='black')
    ax.add_patch(resnet_box)
    ax.text(7.6, 11.9, 'ResNet-101\nBackbone', fontsize=9, ha='center', va='center', fontweight='bold')
    
    rpn_box = Rectangle((9, 11.6), 2.2, 0.6, facecolor='white', alpha=0.9, edgecolor='black')
    ax.add_patch(rpn_box)
    ax.text(10.1, 11.9, 'Region Proposal\nNetwork', fontsize=9, ha='center', va='center', fontweight='bold')
    
    classifier_box = Rectangle((11.5, 11.6), 2.2, 0.6, facecolor='white', alpha=0.9, edgecolor='black')
    ax.add_patch(classifier_box)
    ax.text(12.6, 11.9, '634 Category\nClassifier', fontsize=9, ha='center', va='center', fontweight='bold')
    
    ax.text(10, 11.2, 'Faster R-CNN + Detectron2 Framework', 
            fontsize=10, ha='center', va='center', style='italic')
    
    # Layer 3: Validation Interface
    validation_box = FancyBboxPatch((15.5, 11), 4, 2, boxstyle="round,pad=0.1", 
                                    facecolor=colors['validation'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(validation_box)
    ax.text(17.5, 12, 'VALIDATION\nINTERFACE', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(17.5, 11.5, '• Streamlit Web App\n• Expert Review\n• Real-time Feedback', 
            fontsize=10, ha='center', va='center')
    
    # Layer 4: Database Integration
    db_left = FancyBboxPatch((2, 7.5), 3.5, 2.5, boxstyle="round,pad=0.1", 
                             facecolor=colors['database'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(db_left)
    ax.text(3.75, 9.2, 'TLA DATABASE', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(3.75, 8.5, '• Thesaurus Linguae\n  Aegyptiae\n• Transliterations\n• Translations\n• Linguistic Context', 
            fontsize=9, ha='center', va='center')
    
    db_right = FancyBboxPatch((14.5, 7.5), 3.5, 2.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['database'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(db_right)
    ax.text(16.25, 9.2, 'AKU DATABASE', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(16.25, 8.5, '• Altägyptische\n  Kursivschriften\n• Paleographic Refs\n• Sign Variants\n• SVG Images', 
            fontsize=9, ha='center', va='center')
    
    # Layer 5: Output Layer
    output_box = FancyBboxPatch((8, 4), 4, 2.5, boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10, 5.7, 'RESEARCH OUTPUT', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(10, 4.8, '• Validated Predictions\n• CSV Export\n• Statistical Analysis\n• Academic Publications', 
            fontsize=10, ha='center', va='center')
    
    # Add arrows showing data flow
    # Input to AI
    ax.arrow(4.5, 12, 1.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # AI to Validation
    ax.arrow(14, 12, 1.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Validation to Databases (bidirectional)
    ax.arrow(16, 11, -10, -1, head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.7)
    ax.arrow(17, 11, 0, -1, head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.7)
    
    # To Output
    ax.arrow(15, 10, -2.5, -3.2, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Add data flow labels
    ax.text(5.2, 12.5, 'Raw Images', fontsize=9, ha='center', rotation=0, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(14.7, 12.5, 'Predictions', fontsize=9, ha='center', rotation=0, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(10, 7.5, 'Context\nLookup', fontsize=9, ha='center', rotation=0, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(13, 7, 'Validated\nResults', fontsize=9, ha='center', rotation=-35, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add performance metrics box
    metrics_box = FancyBboxPatch((0.5, 1), 6, 2.5, boxstyle="round,pad=0.1", 
                                 facecolor='lightgray', alpha=0.8, edgecolor='black', linewidth=1)
    ax.add_patch(metrics_box)
    ax.text(3.5, 2.7, 'PERFORMANCE METRICS', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(3.5, 2, '• mAP: 31.2%\n• Detection Rate: 62.1%\n• 634 Gardiner Categories\n• 100% TLA Coverage', 
            fontsize=9, ha='center', va='center', fontfamily='monospace')
    
    # Add technology stack box
    tech_box = FancyBboxPatch((13.5, 1), 6, 2.5, boxstyle="round,pad=0.1", 
                              facecolor='lightgray', alpha=0.8, edgecolor='black', linewidth=1)
    ax.add_patch(tech_box)
    ax.text(16.5, 2.7, 'TECHNOLOGY STACK', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(16.5, 2, '• PyTorch + Detectron2\n• Streamlit Interface\n• OpenCV Processing\n• COCO Format', 
            fontsize=9, ha='center', va='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("docs/system_architecture.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved system architecture: {output_path}")
    
    return fig

def create_validation_workflow_diagram():
    """Create a detailed validation workflow visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    fig.suptitle('HieraticAI Validation Workflow', fontsize=22, fontweight='bold', y=0.95)
    
    # Define workflow steps
    steps = [
        {"pos": (2, 10), "size": (3, 1.5), "title": "1. LOAD PREDICTIONS", 
         "content": "• AI Model Results\n• COCO Format\n• Confidence Scores"},
        
        {"pos": (7, 10), "size": (3, 1.5), "title": "2. FILTER BY THRESHOLD", 
         "content": "• Adjustable Slider\n• Confidence ≥ 0.3\n• Focus on Uncertain"},
        
        {"pos": (12, 10), "size": (3, 1.5), "title": "3. SELECT PREDICTION", 
         "content": "• Click Bounding Box\n• View Cropped Sign\n• Show Metadata"},
        
        {"pos": (2, 7.5), "size": (3, 1.5), "title": "4. LOAD CONTEXT", 
         "content": "• TLA Linguistic Data\n• AKU References\n• Gardiner Info"},
        
        {"pos": (7, 7.5), "size": (3, 1.5), "title": "5. EXPERT REVIEW", 
         "content": "• Compare References\n• Check Accuracy\n• Apply Knowledge"},
        
        {"pos": (12, 7.5), "size": (3, 1.5), "title": "6. MAKE DECISION", 
         "content": "• [OK] Correct\n• [X] Incorrect\n• [?] Uncertain"},
        
        {"pos": (2, 5), "size": (3, 1.5), "title": "7. UPDATE DATABASE", 
         "content": "• Store Validation\n• Track Progress\n• Calculate Stats"},
        
        {"pos": (7, 5), "size": (3, 1.5), "title": "8. SHOW PROGRESS", 
         "content": "• Real-time Stats\n• Progress Bar\n• Accuracy Metrics"},
        
        {"pos": (12, 5), "size": (3, 1.5), "title": "9. EXPORT RESULTS", 
         "content": "• CSV Download\n• Research Data\n• Academic Use"}
    ]
    
    # Draw workflow steps
    for i, step in enumerate(steps):
        x, y = step["pos"]
        w, h = step["size"]
        
        # Determine color based on step type
        if i < 3:  # Input steps
            color = '#3498db'
        elif i < 6:  # Processing steps
            color = '#27ae60'
        else:  # Output steps
            color = '#9b59b6'
        
        # Draw box
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1", 
                             facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Add title
        ax.text(x, y+0.4, step["title"], fontsize=11, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Add content
        ax.text(x, y-0.2, step["content"], fontsize=9, 
                ha='center', va='center', color='white')
    
    # Add workflow arrows
    arrow_pairs = [
        (0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8)
    ]
    
    for start, end in arrow_pairs:
        start_pos = steps[start]["pos"]
        end_pos = steps[end]["pos"]
        
        # Calculate arrow positions
        if start_pos[1] == end_pos[1]:  # Same row
            if start_pos[0] < end_pos[0]:  # Left to right
                sx = start_pos[0] + 1.5
                sy = start_pos[1]
                ex = end_pos[0] - 1.5
                ey = end_pos[1]
            else:  # Right to left
                sx = start_pos[0] - 1.5
                sy = start_pos[1]
                ex = end_pos[0] + 1.5
                ey = end_pos[1]
        else:  # Different rows
            sx = start_pos[0]
            sy = start_pos[1] - 0.75
            ex = end_pos[0]
            ey = end_pos[1] + 0.75
        
        # Draw arrow
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add decision loop arrow (from decision back to select)
    ax.annotate('', xy=(12, 9.25), xytext=(12, 8.25),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
    ax.text(13.5, 8.75, 'Next Sign', fontsize=9, ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # Add interface preview
    interface_box = FancyBboxPatch((0.5, 1.5), 16, 2.5, boxstyle="round,pad=0.2", 
                                   facecolor='lightgray', alpha=0.3, edgecolor='black', linewidth=1)
    ax.add_patch(interface_box)
    ax.text(8.5, 3.5, 'STREAMLIT VALIDATION INTERFACE', fontsize=14, fontweight='bold', ha='center')
    
    # Interface elements
    ax.text(2.5, 2.8, 'MANUSCRIPT\nVIEWER', fontsize=10, fontweight='bold', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#3498db', alpha=0.7))
    
    ax.text(8.5, 2.8, 'SIGN REVIEW\nPANEL', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#27ae60', alpha=0.7))
    
    ax.text(14.5, 2.8, 'VALIDATION\nCONTROLS', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#e74c3c', alpha=0.7))
    
    ax.text(8.5, 2, 'Real-time statistics • Progress tracking • CSV export', 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("docs/validation_workflow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved validation workflow: {output_path}")
    
    return fig

def create_database_integration_diagram():
    """Create a database integration architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    fig.suptitle('Database Integration Architecture', fontsize=20, fontweight='bold', y=0.95)
    
    # Central Gardiner Code
    center_circle = Circle((8, 5), 1.5, facecolor='gold', alpha=0.8, edgecolor='black', linewidth=3)
    ax.add_patch(center_circle)
    ax.text(8, 5, 'GARDINER\nCODE\n(A1, M17, etc.)', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # TLA Database
    tla_box = FancyBboxPatch((1, 7), 4, 2.5, boxstyle="round,pad=0.2", 
                             facecolor='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(tla_box)
    ax.text(3, 8.7, 'TLA DATABASE', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(3, 8, 'Thesaurus Linguae Aegyptiae', fontsize=10, ha='center', color='white', style='italic')
    ax.text(3, 7.4, '• Transliterations\n• Translations\n• Frequency Data\n• Lemma Relations', 
            fontsize=9, ha='center', color='white')
    
    # AKU Database  
    aku_box = FancyBboxPatch((11, 7), 4, 2.5, boxstyle="round,pad=0.2", 
                             facecolor='#f39c12', alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(aku_box)
    ax.text(13, 8.7, 'AKU DATABASE', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(13, 8, 'Altägyptische Kursivschriften', fontsize=10, ha='center', color='white', style='italic')
    ax.text(13, 7.4, '• Paleographic Refs\n• Sign Variants\n• SVG Images\n• Dating Info', 
            fontsize=9, ha='center', color='white')
    
    # Validation Interface
    interface_box = FancyBboxPatch((6, 1), 4, 2, boxstyle="round,pad=0.2", 
                                   facecolor='#27ae60', alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(interface_box)
    ax.text(8, 2.3, 'VALIDATION INTERFACE', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(8, 1.6, '• Expert Review\n• Context Display\n• Decision Making', 
            fontsize=10, ha='center', color='white')
    
    # Fallback Strategy
    fallback_box = FancyBboxPatch((1, 2), 3.5, 2, boxstyle="round,pad=0.1", 
                                  facecolor='lightcoral', alpha=0.7, edgecolor='black', linewidth=1)
    ax.add_patch(fallback_box)
    ax.text(2.75, 3.3, 'FALLBACK STRATEGY', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.75, 2.6, '1. Direct Match\n2. Similar Sign\n3. Manual Entry\n4. Basic Classification', 
            fontsize=9, ha='center')
    
    # Coverage Guarantee
    coverage_box = FancyBboxPatch((11.5, 2), 3.5, 2, boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', alpha=0.7, edgecolor='black', linewidth=1)
    ax.add_patch(coverage_box)
    ax.text(13.25, 3.3, '100% COVERAGE', fontsize=11, fontweight='bold', ha='center')
    ax.text(13.25, 2.6, 'Guaranteed linguistic\ndata for every\ndetected sign', 
            fontsize=9, ha='center')
    
    # Add connecting arrows
    # TLA to center
    ax.annotate('', xy=(7, 6), xytext=(4.5, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax.text(5.5, 6.8, 'Linguistic\nContext', fontsize=9, ha='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # AKU to center
    ax.annotate('', xy=(9, 6), xytext=(11.5, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    ax.text(10.5, 6.8, 'Paleographic\nReferences', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Center to interface
    ax.annotate('', xy=(8, 3.2), xytext=(8, 3.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(9, 3.8, 'Combined\nData', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    # Fallback connections
    ax.annotate('', xy=(6.5, 4.5), xytext=(4.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='dashed'))
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("docs/database_integration.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved database integration: {output_path}")
    
    return fig

def create_research_methodology_diagram():
    """Create research methodology and academic workflow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    fig.suptitle('HieraticAI: Academic Research Methodology', fontsize=22, fontweight='bold', y=0.95)
    
    # Research phases
    phases = [
        {"pos": (3, 10), "size": (4, 1.8), "title": "PHASE 1: DATA PREPARATION", 
         "color": "#3498db", "content": "• Westcar Papyrus digitization\n• CVAT annotation workflow\n• 634 Gardiner categories\n• Spatial data splitting"},
        
        {"pos": (9, 10), "size": (4, 1.8), "title": "PHASE 2: MODEL TRAINING", 
         "color": "#e74c3c", "content": "• Faster R-CNN architecture\n• ResNet-101 backbone\n• Category mapping fix\n• Performance optimization"},
        
        {"pos": (15, 10), "size": (4, 1.8), "title": "PHASE 3: VALIDATION", 
         "color": "#27ae60", "content": "• Expert validation interface\n• TLA/AKU integration\n• Statistical analysis\n• Academic review"},
        
        {"pos": (3, 6.5), "size": (4, 1.8), "title": "PHASE 4: EVALUATION", 
         "color": "#f39c12", "content": "• Performance metrics\n• Comparative analysis\n• Error characterization\n• Methodology validation"},
        
        {"pos": (9, 6.5), "size": (4, 1.8), "title": "PHASE 5: DISSEMINATION", 
         "color": "#9b59b6", "content": "• Academic publication\n• Open-source release\n• Documentation\n• Community engagement"},
        
        {"pos": (15, 6.5), "size": (4, 1.8), "title": "PHASE 6: IMPACT", 
         "color": "#1abc9c", "content": "• Digital humanities\n• Egyptological research\n• Paleographic studies\n• Educational applications"}
    ]
    
    # Draw phases
    for i, phase in enumerate(phases):
        x, y = phase["pos"]
        w, h = phase["size"]
        
        # Draw box
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15", 
                             facecolor=phase["color"], alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Add phase number
        circle = Circle((x-w/2+0.4, y+h/2-0.4), 0.3, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x-w/2+0.4, y+h/2-0.4, str(i+1), fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Add title
        ax.text(x, y+0.5, phase["title"], fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Add content
        ax.text(x, y-0.2, phase["content"], fontsize=10, 
                ha='center', va='center', color='white')
    
    # Add flow arrows
    arrow_pairs = [(0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)]
    
    for start, end in arrow_pairs:
        start_pos = phases[start]["pos"]
        end_pos = phases[end]["pos"]
        
        # Calculate arrow direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Normalize and adjust for box boundaries
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            
            sx = start_pos[0] + dx_norm * 2.2
            sy = start_pos[1] + dy_norm * 1.0
            ex = end_pos[0] - dx_norm * 2.2
            ey = end_pos[1] - dy_norm * 1.0
            
            ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
    
    # Add methodology details
    method_box = FancyBboxPatch((1, 2.5), 16, 2.5, boxstyle="round,pad=0.2", 
                                facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1)
    ax.add_patch(method_box)
    
    ax.text(9, 4.5, 'METHODOLOGICAL CONTRIBUTIONS', fontsize=16, fontweight='bold', ha='center')
    
    # Three columns of contributions
    ax.text(4, 3.7, 'TECHNICAL INNOVATIONS', fontsize=12, fontweight='bold', ha='center', color='darkblue')
    ax.text(4, 3.2, '• Category mapping fix discovery\n• Spatial data splitting strategy\n• Multi-database integration\n• Real-time validation interface', 
            fontsize=10, ha='center')
    
    ax.text(9, 3.7, 'ACADEMIC RIGOR', fontsize=12, fontweight='bold', ha='center', color='darkgreen')
    ax.text(9, 3.2, '• Expert validation methodology\n• Comprehensive error analysis\n• Statistical significance testing\n• Reproducible research practices', 
            fontsize=10, ha='center')
    
    ax.text(14, 3.7, 'IMPACT & APPLICATIONS', fontsize=12, fontweight='bold', ha='center', color='darkred')
    ax.text(14, 3.2, '• Digital paleography advancement\n• Egyptological tool development\n• Open science contribution\n• Educational resource creation', 
            fontsize=10, ha='center')
    
    # Add FU Berlin attribution
    fu_box = FancyBboxPatch((6, 0.3), 6, 1, boxstyle="round,pad=0.1", 
                            facecolor='lightblue', alpha=0.8, edgecolor='navy', linewidth=2)
    ax.add_patch(fu_box)
    ax.text(9, 0.8, 'Freie Universität Berlin • Ancient Language Processing Seminar', 
            fontsize=11, fontweight='bold', ha='center')
    ax.text(9, 0.5, 'Winter 2025 • Methodological Prototype', 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("docs/research_methodology.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved research methodology: {output_path}")
    
    return fig

def main():
    """Generate all workflow and architecture visualizations."""
    print("reating comprehensive workflow visualizations...")
    print("="* 60)
    
    try:
        # Create output directory
        Path("docs").mkdir(exist_ok=True)
        
        # Generate all visualizations
        print("\nGenerating visualizations:")
        fig1 = create_system_architecture_diagram()
        fig2 = create_validation_workflow_diagram()
        fig3 = create_database_integration_diagram()
        fig4 = create_research_methodology_diagram()
        
        print("\nAll workflow visualizations created successfully!")
        print("\nGenerated files:")
        print("- docs/system_architecture.png")
        print("- docs/validation_workflow.png") 
        print("- docs/database_integration.png")
        print("- docs/research_methodology.png")
        
        # Show plots if running interactively
        try:
            plt.show()
        except:
            pass  # In case we're running in a non-interactive environment
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nWorkflow visualization generation completed successfully!")
    else:
        print("\nWorkflow visualization generation failed!")
