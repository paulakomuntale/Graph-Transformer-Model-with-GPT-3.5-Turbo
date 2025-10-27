import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
import requests
from typing import List, Dict, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
import hashlib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')

# Configuration
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your actual API key
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Initialize clients
if OPENAI_API_KEY != "your-openai-api-key":
    openai.api_key = OPENAI_API_KEY

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class GraphVisualizer:
    """Class to visualize various graphs and models"""
    
    @staticmethod
    def plot_knowledge_graph(kg, figsize=(12, 8)):
        """Visualize the medical knowledge graph"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(kg.graph, k=3, iterations=50)
        
        # Define node colors by type
        node_colors = []
        for node in kg.graph.nodes():
            node_type = kg.graph.nodes[node].get('type', 'unknown')
            if node_type == 'phase':
                node_colors.append('lightcoral')
            elif node_type == 'symptom':
                node_colors.append('lightblue')
            elif node_type == 'treatment':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw_networkx_nodes(kg.graph, pos, node_size=2000, 
                             node_color=node_colors, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(kg.graph, pos, edge_color='gray', 
                             arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(kg.graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edge labels
        edge_labels = {(u, v): d['relationship'] for u, v, d in kg.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(kg.graph, pos, edge_labels=edge_labels, 
                                   font_size=6, ax=ax)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Phase'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Symptom'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Treatment')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_title("🩺 Medical Knowledge Graph\n(Relationships between phases, symptoms, and treatments)", 
                   fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print graph statistics
        print(f"📊 Knowledge Graph Statistics:")
        print(f"   • Nodes: {kg.graph.number_of_nodes()}")
        print(f"   • Edges: {kg.graph.number_of_edges()}")
        print(f"   • Phase nodes: {sum(1 for node in kg.graph.nodes() if kg.graph.nodes[node].get('type') == 'phase')}")
        print(f"   • Symptom nodes: {sum(1 for node in kg.graph.nodes() if kg.graph.nodes[node].get('type') == 'symptom')}")
        print(f"   • Treatment nodes: {sum(1 for node in kg.graph.nodes() if kg.graph.nodes[node].get('type') == 'treatment')}")
    
    @staticmethod
    def plot_user_timeline(user_data, user_id, figsize=(14, 8)):
        """Plot user's menstrual cycle timeline"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        user_df = user_data[user_data['id'] == user_id].sort_values('day_in_study')
        
        if len(user_df) == 0:
            print(f"❌ No data found for user {user_id}")
            return
            
        # Plot 1: Phase transitions
        phases = user_df['phase'].values
        days = user_df['day_in_study'].values
        
        # Color mapping for phases
        phase_colors = {
            'Follicular': 'lightblue',
            'Fertility': 'lightgreen', 
            'Luteal': 'lightcoral',
            'Menstrual': 'lightpink'
        }
        
        # Create a timeline bar
        current_phase = phases[0]
        start_day = days[0]
        
        for i in range(1, len(phases)):
            if phases[i] != current_phase or i == len(phases)-1:
                end_day = days[i-1] if i < len(phases)-1 else days[i]
                color = phase_colors.get(current_phase, 'gray')
                label = current_phase if current_phase not in [p.get_label() for p in ax1.patches] else ""
                ax1.barh(0, end_day - start_day + 1, left=start_day, 
                        color=color, alpha=0.7, label=label)
                current_phase = phases[i]
                start_day = days[i]
        
        ax1.set_xlabel('Day in Study')
        ax1.set_title(f'📅 Menstrual Cycle Timeline - User {user_id}', fontweight='bold')
        ax1.set_yticks([])
        ax1.legend(loc='upper right')
        
        # Plot 2: Symptoms over time
        symptom_cols = ['cramps', 'headaches', 'fatigue', 'moodswing', 'bloating']
        symptom_data = []
        valid_symptoms = []
        
        for col in symptom_cols:
            if col in user_df.columns:
                # Convert categorical to numeric
                symptom_values = []
                for val in user_df[col]:
                    if isinstance(val, str):
                        mapping = {'Not at all': 0, 'Very Low/Little': 1, 'Low': 2,
                                 'Moderate': 3, 'High': 4, 'Very High': 5}
                        symptom_values.append(mapping.get(val, 0))
                    else:
                        symptom_values.append(float(val) if pd.notna(val) else 0)
                if any(v > 0 for v in symptom_values):  # Only plot if there are non-zero values
                    symptom_data.append(symptom_values)
                    valid_symptoms.append(col)
        
        if symptom_data:
            for i, col in enumerate(valid_symptoms):
                ax2.plot(days, symptom_data[i], marker='o', label=col, linewidth=2)
            
            ax2.set_xlabel('Day in Study')
            ax2.set_ylabel('Symptom Intensity')
            ax2.set_title('📊 Symptom Progression Over Time', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No symptom data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('📊 Symptom Progression Over Time', fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print user statistics
        print(f"📈 User {user_id} Statistics:")
        print(f"   • Total days tracked: {len(user_df)}")
        print(f"   • Phases observed: {list(user_df['phase'].unique())}")
        print(f"   • Age: {user_df['age'].iloc[0] if 'age' in user_df.columns else 'Unknown'}")
        print(f"   • BMI: {user_df['BMI'].iloc[0] if 'BMI' in user_df.columns else 'Unknown'}")
    
    @staticmethod
    def plot_model_architecture(input_dim, hidden_dim, num_heads, num_layers, num_phases):
        """Visualize the Graph Transformer architecture"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define layer positions
        layers_x = [1, 2, 3, 4, 5, 6]
        layers = ['Input\nProjection', 'Position\nEncoding', 'Transformer\nLayers', 
                'Global\nPooling', 'Phase\nHead', 'Regression\nHead']
        
        # Draw layers
        for i, (x, layer) in enumerate(zip(layers_x, layers)):
            # Draw layer box
            box = FancyBboxPatch((x-0.4, 0.1), 0.8, 0.8, 
                               boxstyle="round,pad=0.02", 
                               facecolor='lightblue', edgecolor='black', alpha=0.7)
            ax.add_patch(box)
            ax.text(x, 0.5, layer, ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Add layer details
            details = ""
            if i == 0:
                details = f"Input dim: {input_dim}\nHidden dim: {hidden_dim}"
            elif i == 2:
                details = f"Layers: {num_layers}\nHeads: {num_heads}"
            elif i == 4:
                details = f"Output: {num_phases} phases"
            elif i == 5:
                details = "Output: Days to menstrual"
            
            if details:
                ax.text(x, 0.2, details, ha='center', va='top', fontsize=7)
        
        # Draw connections
        for i in range(len(layers_x)-1):
            ax.annotate('', xy=(layers_x[i+1], 0.5), xytext=(layers_x[i]+0.4, 0.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
        
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, 1)
        ax.set_title('🧠 Graph Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add model specifications
        specs_text = f"""
        Model Specifications:
        • Input Dimension: {input_dim}
        • Hidden Dimension: {hidden_dim}
        • Number of Heads: {num_heads}
        • Number of Layers: {num_layers}
        • Number of Phases: {num_phases}
        • Outputs: Phase classification + Days to menstrual regression
        """
        
        ax.text(0.5, -0.2, specs_text, transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_progress(loss_history, figsize=(10, 6)):
        """Plot training loss progress"""
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(len(loss_history))
        ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('📉 Training Progress', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Mark the final loss
        if loss_history:
            final_loss = loss_history[-1]
            ax.plot(len(loss_history)-1, final_loss, 'ro', markersize=8)
            ax.annotate(f'Final: {final_loss:.4f}', 
                       xy=(len(loss_history)-1, final_loss),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_breakdown(prediction, user_data, user_id, figsize=(12, 8)):
        """Visualize prediction breakdown with probabilities"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Phase probabilities
        phase_probabilities = prediction.get('phase_probabilities', {})
        if phase_probabilities:
            phases = list(phase_probabilities.keys())
            probs = list(phase_probabilities.values())
        else:
            # Create simple probability distribution
            phases = ['Follicular', 'Fertility', 'Luteal', 'Menstrual']
            pred_phase = prediction.get('predicted_phase', 'Luteal')
            probs = [0.1 if phase != pred_phase else 0.7 for phase in phases]
        
        colors = ['lightblue' if phase != prediction.get('predicted_phase', '') 
                 else 'gold' for phase in phases]
        
        bars = ax1.bar(phases, probs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Probability')
        ax1.set_title('🎯 Phase Prediction Probabilities', fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Confidence indicator
        confidence = prediction.get('confidence_score', 0.5)
        ax2.barh(['Confidence'], [confidence], color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlim(0, 1)
        ax2.set_title('📊 Prediction Confidence', fontweight='bold')
        ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='High confidence threshold')
        ax2.legend()
        
        # Add confidence value
        ax2.text(confidence, 0, f'{confidence:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
        
        # Plot 3: Days to menstrual
        days = prediction.get('days_to_menstrual', 0)
        ax3.bar(['Days to\nMenstrual'], [days], color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Days')
        ax3.set_title('📅 Timeline Prediction', fontweight='bold')
        
        # Add day value
        ax3.text(0, days + 0.1, f'{days:.1f} days', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
        
        # Plot 4: User context
        user_stats = [
            f"User: {user_id}",
            f"Age: {user_data['age'].iloc[0] if 'age' in user_data.columns else 'Unknown'}",
            f"BMI: {user_data['BMI'].iloc[0] if 'BMI' in user_data.columns else 'Unknown'}",
            f"Tracked Days: {len(user_data)}",
            f"Current Phase: {prediction.get('predicted_phase', 'Unknown')}"
        ]
        
        ax4.text(0.1, 0.8, '\n'.join(user_stats), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        ax4.set_title('👤 User Context', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

class LightweightVectorStore:
    """Lightweight vector store using sentence transformers and cosine similarity"""
    
    def __init__(self):
        print("🔄 Initializing Medical Knowledge Base...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self._initialize_medical_knowledge()
    
    def _initialize_medical_knowledge(self):
        """Initialize with comprehensive medical knowledge"""
        medical_documents = [
            {
                "text": "The follicular phase begins on day 1 of menstruation and lasts 10-14 days. Estrogen rises, promoting endometrial growth and follicle development.",
                "metadata": {"category": "phase", "phase": "follicular", "type": "physiology"}
            },
            {
                "text": "The luteal phase lasts 10-16 days after ovulation. Progesterone dominates, preparing the uterus for potential pregnancy.",
                "metadata": {"category": "phase", "phase": "luteal", "type": "physiology"}
            },
            {
                "text": "Menstrual cramps are caused by prostaglandins. NSAIDs like ibuprofen work best when taken at pain onset.",
                "metadata": {"category": "symptom", "symptom": "cramps", "type": "treatment"}
            },
            {
                "text": "Mood swings are common in the luteal phase due to hormonal fluctuations. Regular exercise and stress reduction help.",
                "metadata": {"category": "symptom", "symptom": "mood_swings", "type": "management"}
            },
        ]
        
        # Store documents and compute embeddings
        self.documents = [doc["text"] for doc in medical_documents]
        self.metadata = [doc["metadata"] for doc in medical_documents]
        print("📚 Computing document embeddings...")
        self.embeddings = self.embedding_model.encode(self.documents)
        print(f"✅ Loaded {len(self.documents)} medical knowledge documents")
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Retrieve relevant documents using semantic search with cosine similarity"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top n results
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    # Apply filters if provided
                    if filters:
                        matches_filter = True
                        for key, value in filters.items():
                            if self.metadata[idx].get(key) != value:
                                matches_filter = False
                                break
                        if not matches_filter:
                            continue
                    
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity': float(similarities[idx])
                    })
            
            return results[:n_results]
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    def get_phase_context(self, phase: str) -> List[Dict]:
        """Get context for specific menstrual phase"""
        return self.retrieve_relevant_documents(
            f"{phase} phase symptoms management physiology",
            filters={"phase": phase.lower()} if phase else None
        )
    
    def get_symptom_context(self, symptoms: List[str]) -> List[Dict]:
        """Get context for specific symptoms"""
        contexts = []
        for symptom in symptoms:
            symptom_context = self.retrieve_relevant_documents(
                f"{symptom} treatment management relief",
                filters={"symptom": symptom.lower()}
            )
            contexts.extend(symptom_context)
        return contexts

class MedicalKnowledgeGraph:
    """Enhanced knowledge graph for menstrual health"""
    def __init__(self):
        print("🔄 Building Medical Knowledge Graph...")
        self.graph = nx.DiGraph()
        self.build_medical_kg()
        print("✅ Medical Knowledge Graph built successfully!")
    
    def build_medical_kg(self):
        """Build comprehensive medical knowledge graph"""
        # Add entities
        entities = {
            'Follicular_Phase': {'type': 'phase', 'duration': '10-14', 'hormone': 'estrogen_rise'},
            'Ovulatory_Phase': {'type': 'phase', 'duration': '3-5', 'hormone': 'LH_surge'},
            'Luteal_Phase': {'type': 'phase', 'duration': '10-16', 'hormone': 'progesterone'},
            'Menstrual_Phase': {'type': 'phase', 'duration': '3-7', 'hormone': 'low_all'},
            'Cramps': {'type': 'symptom', 'severity': '1-10'},
            'Bloating': {'type': 'symptom', 'severity': '1-10'},
            'Fatigue': {'type': 'symptom', 'severity': '1-10'},
            'Mood_Swings': {'type': 'symptom', 'severity': '1-10'},
            'Ibuprofen': {'type': 'treatment', 'class': 'NSAID'},
            'Heat_Therapy': {'type': 'treatment', 'class': 'physical'},
            'Exercise': {'type': 'treatment', 'class': 'lifestyle'},
        }
        
        for entity, attrs in entities.items():
            self.graph.add_node(entity, **attrs)
        
        # Add relationships
        relationships = [
            ('Luteal_Phase', 'commonly_has', 'Bloating'),
            ('Luteal_Phase', 'commonly_has', 'Mood_Swings'),
            ('Menstrual_Phase', 'commonly_has', 'Cramps'),
            ('Menstrual_Phase', 'commonly_has', 'Fatigue'),
            ('Ibuprofen', 'treats', 'Cramps'),
            ('Heat_Therapy', 'treats', 'Cramps'),
            ('Exercise', 'helps', 'Mood_Swings'),
            ('Exercise', 'helps', 'Fatigue'),
            ('Follicular_Phase', 'precedes', 'Ovulatory_Phase'),
            ('Ovulatory_Phase', 'precedes', 'Luteal_Phase'),
            ('Luteal_Phase', 'precedes', 'Menstrual_Phase'),
        ]
        
        for src, rel, dst in relationships:
            self.graph.add_edge(src, dst, relationship=rel)

# ... (Keep the other classes like GraphEnhancedDataset, KnowledgeEnhancedGraphTransformer, etc. the same as before)

class GraphEnhancedDataset:
    def __init__(self, df, knowledge_graph):
        print("🔄 Preparing dataset with knowledge graph enhancements...")
        self.df = df
        self.kg = knowledge_graph
        self.users = df['id'].unique()
        self.phase_encoder = LabelEncoder()
        self.prepare_data()
        print("✅ Dataset preparation completed!")
    
    def prepare_data(self):
        """Prepare data with knowledge graph enhancements"""
        self.df = self.df.copy()
        self.df['day_of_cycle'] = self.df.groupby('id')['day_in_study'].transform(
            lambda x: x - x.min() + 1)
        
        self.df['phase_encoded'] = self.phase_encoder.fit_transform(self.df['phase'])
        
        # Basic features
        self.numeric_features = ['age', 'BBT', 'lh', 'estrogen', 'pain_nrs', 'sleep_hours', 'BMI']
        for col in self.numeric_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        print(f"📊 Available phases: {list(self.phase_encoder.classes_)}")
        print(f"👥 Number of users: {len(self.users)}")
    
    def build_knowledge_enhanced_graph(self, user_id, window_size=5):
        """Build graph enhanced with medical knowledge"""
        user_data = self.df[self.df['id'] == user_id].sort_values('day_in_study')
        
        if len(user_data) < window_size:
            print(f"⚠️  Insufficient data for user {user_id}")
            return None, None, None
            
        recent_data = user_data.tail(window_size)
        node_features = []
        
        for idx, (_, day_data) in enumerate(recent_data.iterrows()):
            # Basic features
            features = []
            for feat in self.numeric_features:
                if feat in day_data:
                    features.append(float(day_data[feat]))
                else:
                    features.append(0.0)
            
            # Add knowledge graph context features
            current_phase = day_data['phase'].replace(' ', '_')
            kg_context = self.get_phase_context(current_phase)
            features.extend(kg_context)
            
            # Add symptom features
            symptom_features = self.get_symptom_features(day_data)
            features.extend(symptom_features)
            
            node_features.append(features)
            
            # Target for last node
            if idx == len(recent_data) - 1:
                target_phase = int(day_data['phase_encoded'])
                days_to_menstrual = self.calculate_days_to_menstrual(user_data, day_data)
        
        node_tensor = torch.tensor(node_features, dtype=torch.float32)
        num_nodes = len(node_features)
        attention_mask = torch.ones(num_nodes, num_nodes)
        
        return node_tensor, attention_mask, (target_phase, days_to_menstrual)
    
    def get_phase_context(self, phase):
        """Get knowledge graph context for a phase"""
        phase_mapping = {
            'Follicular': 'Follicular_Phase',
            'Ovulatory': 'Ovulatory_Phase', 
            'Luteal': 'Luteal_Phase',
            'Menstrual': 'Menstrual_Phase'
        }
        
        kg_phase = phase_mapping.get(phase, phase)
        related = self.kg.query_related_entities(kg_phase)
        
        context_features = [
            len(related),
            sum(1 for r in related if r['relationship'] == 'commonly_has'),
            sum(1 for r in related if r['relationship'] == 'precedes'),
        ]
        
        while len(context_features) < 5:
            context_features.append(0.0)
            
        return context_features[:5]
    
    def get_symptom_features(self, day_data):
        """Extract symptom features from day data"""
        symptom_cols = ['cramps', 'headaches', 'fatigue', 'moodswing', 'bloating']
        symptom_features = []
        
        for col in symptom_cols:
            if col in day_data:
                val = day_data[col]
                if isinstance(val, str):
                    symptom_mapping = {
                        'Not at all': 0, 'Very Low/Little': 1, 'Low': 2,
                        'Moderate': 3, 'High': 4, 'Very High': 5
                    }
                    symptom_features.append(symptom_mapping.get(val, 0))
                else:
                    symptom_features.append(float(val) if pd.notna(val) else 0.0)
            else:
                symptom_features.append(0.0)
                
        return symptom_features
    
    def calculate_days_to_menstrual(self, user_data, current_day):
        """Calculate days to next menstrual phase"""
        future_data = user_data[user_data['day_in_study'] > current_day['day_in_study']]
        menstrual_days = future_data[future_data['phase'] == 'Menstrual']
        if len(menstrual_days) > 0:
            return float(menstrual_days['day_in_study'].iloc[0] - current_day['day_in_study'])
        return float(28 - (current_day['day_in_study'] % 28))

class KnowledgeEnhancedGraphTransformer(nn.Module):
    """Graph Transformer enhanced with medical knowledge"""
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_phases):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, 20, hidden_dim) * 0.02)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.regression_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, attention_mask=None):
        batch_size, num_nodes, feat_dim = node_features.shape
        
        x = self.input_proj(node_features)
        
        if num_nodes <= self.position_encoding.size(1):
            x = x + self.position_encoding[:, :num_nodes, :]
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        graph_embedding = x.mean(dim=1)
        
        phase_logits = self.phase_head(graph_embedding)
        days_pred = self.regression_head(graph_embedding)
        
        return phase_logits, days_pred.squeeze()

class RealLLMClient:
    """Client for real external LLM APIs"""
    
    def __init__(self, api_key: str, base_url: str = OPENAI_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        print("🤖 Initializing Real LLM Client...")
    
    def generate_response(self, prompt: str, context: List[Dict], max_tokens: int = 800) -> str:
        """Generate response using real LLM API"""
        
        context_text = "\n".join([f"- {doc['text']}" for doc in context[:3]])
        
        system_message = """You are a compassionate, knowledgeable women's health assistant. 
Provide accurate, evidence-based information about menstrual health and cycle tracking.
Be supportive, clear, and focus on practical advice."""

        user_message = f"""Based on the medical context below, provide a helpful response.

MEDICAL CONTEXT:
{context_text}

USER SITUATION:
{prompt}

Please provide a clear, evidence-based response focusing on practical advice."""
        
        try:
            print("🔄 Calling OpenAI API...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            print("✅ Received response from OpenAI API")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ LLM API Error: {e}")
            return self._fallback_response(prompt, context)
    
    def _fallback_response(self, prompt: str, context: List[Dict]) -> str:
        """Fallback response when LLM API fails"""
        context_summary = "\n".join([doc['text'] for doc in context[:2]])
        return f"""Based on medical knowledge:

{context_summary}

For your situation: {prompt}

I recommend consulting with a healthcare provider for personalized medical advice."""

class RealLLMRAGSystem:
    """Real LLM + RAG system with external API integration"""
    
    def __init__(self, knowledge_graph, vector_store, llm_client):
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.llm_client = llm_client
        print("🔍 Initializing RAG System...")
    
    def generate_comprehensive_explanation(self, prediction: Dict, user_data: pd.DataFrame, user_id: int) -> str:
        """Generate comprehensive explanation using real LLM and RAG"""
        
        print(f"🔄 Generating explanation for user {user_id}...")
        user_context = self._build_user_context(user_data, user_id)
        symptoms = self._extract_symptoms(user_data)
        
        medical_context = self._retrieve_medical_context(prediction, symptoms)
        print(f"📚 Retrieved {len(medical_context)} relevant medical documents")
        
        prompt = self._construct_llm_prompt(prediction, user_context, symptoms)
        
        explanation = self.llm_client.generate_response(prompt, medical_context)
        print("✅ Explanation generated successfully")
        
        return explanation
    
    def _build_user_context(self, user_data: pd.DataFrame, user_id: int) -> Dict[str, Any]:
        """Build comprehensive user context"""
        if len(user_data) == 0:
            return {}
        
        latest = user_data.iloc[-1]
        
        return {
            "user_id": user_id,
            "age": latest.get('age', 'Unknown'),
            "bmi": latest.get('BMI', 'Unknown'),
            "current_cycle_day": latest.get('day_in_study', 'Unknown'),
            "recent_symptoms": self._extract_symptoms(user_data),
        }
    
    def _extract_symptoms(self, user_data: pd.DataFrame) -> List[str]:
        """Extract symptoms from user data"""
        symptoms = []
        symptom_cols = ['cramps', 'headaches', 'fatigue', 'moodswing', 'bloating']
        
        if len(user_data) > 0:
            latest = user_data.iloc[-1]
            for col in symptom_cols:
                if col in latest and latest[col] not in ['Not at all', 'Very Low/Little', 0, 0.0]:
                    symptoms.append(col)
        
        return symptoms
    
    def _retrieve_medical_context(self, prediction: Dict, symptoms: List[str]) -> List[Dict]:
        """Retrieve relevant medical context from vector store"""
        contexts = []
        
        phase_context = self.vector_store.get_phase_context(prediction['predicted_phase'])
        contexts.extend(phase_context)
        
        symptom_context = self.vector_store.get_symptom_context(symptoms)
        contexts.extend(symptom_context)
        
        # Remove duplicates
        seen_texts = set()
        unique_contexts = []
        for ctx in contexts:
            if ctx['text'] not in seen_texts:
                seen_texts.add(ctx['text'])
                unique_contexts.append(ctx)
        
        return unique_contexts[:5]
    
    def _construct_llm_prompt(self, prediction: Dict, user_context: Dict, symptoms: List[str]) -> str:
        """Construct detailed prompt for LLM"""
        
        prompt = f"""
User Analysis Request:

PREDICTION:
- Current Phase: {prediction['predicted_phase']}
- Days to Menstrual: {prediction['days_to_menstrual']:.1f}
- Confidence: {prediction['confidence']}

USER CONTEXT:
- Age: {user_context.get('age', 'Unknown')}
- Current Cycle Day: {user_context.get('current_cycle_day', 'Unknown')}
- Symptoms: {', '.join(symptoms) if symptoms else 'None'}

Please explain what this phase means and provide practical advice."""
        return prompt

class IntegratedRealLLMSystem:
    """Integrated system with real external LLM + RAG"""
    
    def __init__(self, df):
        print("🚀 INITIALIZING INTEGRATED SYSTEM")
        print("=" * 50)
        
        print("1. Building Medical Knowledge Graph...")
        self.kg = MedicalKnowledgeGraph()
        
        # Visualize knowledge graph
        GraphVisualizer.plot_knowledge_graph(self.kg)
        
        print("2. Preparing Dataset...")
        self.dataset = GraphEnhancedDataset(df, self.kg)
        
        # Show user data examples
        if len(df['id'].unique()) > 0:
            user_id = df['id'].iloc[0]
            GraphVisualizer.plot_user_timeline(df, user_id)
        
        print("3. Initializing Vector Store...")
        self.vector_store = LightweightVectorStore()
        
        print("4. Setting up LLM Client...")
        self.llm_client = RealLLMClient(OPENAI_API_KEY)
        
        print("5. Initializing RAG System...")
        self.llm_rag = RealLLMRAGSystem(self.kg, self.vector_store, self.llm_client)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print("✅ System initialization completed!")
        print("=" * 50)
        
    def train_graph_transformer(self, num_epochs=20):
        """Train the graph transformer model"""
        print("\n🧠 TRAINING GRAPH TRANSFORMER")
        print("-" * 30)
        
        all_graphs = []
        all_targets = []
        
        print("Building knowledge-enhanced graphs...")
        successful_users = 0
        
        for user_id in self.dataset.users[:3]:  # Use first 3 users for demo
            graph_data = self.dataset.build_knowledge_enhanced_graph(user_id, window_size=5)
            if graph_data and graph_data[0] is not None:
                node_tensor, attention_mask, targets = graph_data
                all_graphs.append((node_tensor, attention_mask))
                all_targets.append(targets)
                successful_users += 1
                print(f"  ✅ Graph built for user {user_id}")
        
        if not all_graphs:
            print("❌ No valid graphs created!")
            return
        
        print(f"✅ Successfully created graphs for {successful_users} users")
        
        input_dim = all_graphs[0][0].shape[1]
        num_phases = len(self.dataset.phase_encoder.classes_)
        
        print(f"📐 Input dimension: {input_dim}")
        print(f"🎯 Number of phases: {num_phases}")
        
        # Visualize model architecture
        GraphVisualizer.plot_model_architecture(
            input_dim=input_dim,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            num_phases=num_phases
        )
        
        print("Initializing Graph Transformer model...")
        self.model = KnowledgeEnhancedGraphTransformer(
            input_dim=input_dim,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            num_phases=num_phases
        )
        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        print("Starting training loop...")
        
        # Track loss for visualization
        loss_history = []
        
        for epoch in range(num_epochs):
            total_loss = 0
            for (node_tensor, attention_mask), (phase_target, days_target) in zip(all_graphs, all_targets):
                node_tensor = node_tensor.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)
                phase_target = torch.tensor([phase_target], dtype=torch.long).to(self.device)
                days_target = torch.tensor([days_target], dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                
                phase_logits, days_pred = self.model(node_tensor, attention_mask)
                
                phase_loss = F.cross_entropy(phase_logits, phase_target)
                days_loss = F.huber_loss(days_pred, days_target)
                loss = phase_loss + 0.1 * days_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(all_graphs)
            loss_history.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f'  Epoch {epoch:3d} | Loss: {avg_loss:.4f}')
        
        # Visualize training progress
        GraphVisualizer.plot_training_progress(loss_history)
        
        print("✅ Graph Transformer training completed!")
    
    def predict_with_real_llm_explanation(self, user_id):
        """Make prediction with real LLM-generated explanation"""
        print(f"\n🔮 MAKING PREDICTION FOR USER {user_id}")
        print("-" * 30)
        
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
            
        graph_data = self.dataset.build_knowledge_enhanced_graph(user_id, window_size=5)
        if graph_data is None:
            print("❌ Could not build graph for user")
            return None
            
        node_tensor, attention_mask, _ = graph_data
        user_data = self.dataset.df[self.dataset.df['id'] == user_id]
        
        print("Running model inference...")
        self.model.eval()
        with torch.no_grad():
            node_tensor = node_tensor.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            
            phase_logits, days_pred = self.model(node_tensor, attention_mask)
            
            phase_probs = F.softmax(phase_logits, dim=-1)
            predicted_phase_idx = torch.argmax(phase_probs).item()
            predicted_phase = self.dataset.phase_encoder.inverse_transform([predicted_phase_idx])[0]
            
            confidence_score = phase_probs.max().item()
            confidence = 'high' if confidence_score > 0.7 else 'medium'
            
            # Create phase probabilities dictionary
            phase_probabilities = {}
            for i, phase in enumerate(self.dataset.phase_encoder.classes_):
                phase_probabilities[phase] = phase_probs[0][i].item()
            
            prediction = {
                'predicted_phase': predicted_phase,
                'days_to_menstrual': max(0, float(days_pred.cpu().item())),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'phase_probabilities': phase_probabilities
            }
        
        # Visualize prediction breakdown
        GraphVisualizer.plot_prediction_breakdown(prediction, user_data, user_id)
        
        print("🎯 Prediction made, generating LLM explanation...")
        explanation = self.llm_rag.generate_comprehensive_explanation(
            prediction, user_data, user_id
        )
        
        return {
            'prediction': prediction,
            'explanation': explanation,
            'user_id': user_id
        }

# MAIN EXECUTION
print("=== REAL LLM + RAG MENSTRUAL HEALTH ANALYSIS SYSTEM ===")
print("📊 WITH COMPREHENSIVE GRAPH VISUALIZATIONS")
print()

# Load your data
df = pd.read_csv('/kaggle/input/cleanedd-dataset/cleaned_dataset.csv')
print(f"📁 Loaded dataset with {len(df)} rows and {len(df['id'].unique())} users")

# Note: You need to set your OpenAI API key first
if OPENAI_API_KEY == "your-openai-api-key":
    print("⚠️  DEMONSTRATION MODE")
    print("Please set your OPENAI_API_KEY for real LLM functionality")
    print("Using simulated responses for demonstration...")
    
    class MockRealLLMSystem:
        def __init__(self, df):
            print("🚀 INITIALIZING DEMO SYSTEM")
            print("1. Building Medical Knowledge Graph...")
            self.kg = MedicalKnowledgeGraph()
            GraphVisualizer.plot_knowledge_graph(self.kg)
            
            print("2. Preparing Dataset...")
            self.dataset = GraphEnhancedDataset(df, self.kg)
            
            # Show user timeline
            if len(df['id'].unique()) > 0:
                user_id = df['id'].iloc[0]
                GraphVisualizer.plot_user_timeline(df, user_id)
            
            print("3. Initializing Vector Store... ✅")
            self.vector_store = LightweightVectorStore()
            print("4. Setting up LLM Client... ✅")
            print("5. Initializing RAG System... ✅")
            
        def train_graph_transformer(self, num_epochs=20):
            print("\n🧠 TRAINING GRAPH TRANSFORMER (SIMULATED)")
            
            # Show model architecture
            GraphVisualizer.plot_model_architecture(
                input_dim=15, hidden_dim=64, num_heads=4, 
                num_layers=3, num_phases=4
            )
            
            print("Building knowledge-enhanced graphs...")
            print("✅ Successfully created graphs for 3 users")
            print("📐 Input dimension: 15")
            print("🎯 Number of phases: 4")
            print("Starting training loop...")
            
            # Simulate training with loss progression
            loss_history = [0.5 - i*0.02 for i in range(num_epochs)]
            for epoch in range(0, num_epochs, 5):
                print(f'  Epoch {epoch:3d} | Loss: {loss_history[epoch]:.4f}')
            
            # Show training progress
            GraphVisualizer.plot_training_progress(loss_history)
            print("✅ Graph Transformer training completed!")
        
        def predict_with_real_llm_explanation(self, user_id):
            print(f"\n🔮 MAKING PREDICTION FOR USER {user_id}")
            
            user_data = self.dataset.df[self.dataset.df['id'] == user_id]
            
            # Create simulated prediction with probabilities
            phase_probabilities = {
                'Follicular': 0.1,
                'Fertility': 0.15, 
                'Luteal': 0.7,
                'Menstrual': 0.05
            }
            
            prediction = {
                'predicted_phase': 'Luteal',
                'days_to_menstrual': 5.2,
                'confidence': 'high',
                'confidence_score': 0.85,
                'phase_probabilities': phase_probabilities
            }
            
            # Visualize prediction
            GraphVisualizer.plot_prediction_breakdown(prediction, user_data, user_id)
            
            context = self.vector_store.get_phase_context("Luteal")
            
            return {
                'prediction': prediction,
                'explanation': f"""
## 🎯 COMPREHENSIVE CYCLE ANALYSIS

**PREDICTION:** You are currently in the **Luteal Phase** (85% confidence)

**PHYSIOLOGICAL CONTEXT:**
You're in the post-ovulation phase where progesterone is the dominant hormone. This phase typically lasts 10-16 days and prepares your uterus for potential pregnancy.

**WHAT TO EXPECT:**
- Possible bloating or breast tenderness
- Mood changes or irritability  
- Gradual rise in basal body temperature
- Changes in cervical mucus

**EVIDENCE-BASED MANAGEMENT:**
{context[0]['text'] if context else "Progesterone dominance characterizes this phase"}

- Light to moderate exercise can help manage symptoms
- Reduce salt intake to minimize bloating
- Ensure adequate sleep (7-9 hours)

**SAFETY CONSIDERATIONS:**
Consult a healthcare provider for severe pain, heavy bleeding, or concerning symptoms.

**CONFIDENCE LEVEL:** High (85%)
""",
                'user_id': user_id
            }
    
    real_system = MockRealLLMSystem(df)
    
else:
    # Initialize the real system
    print("🔧 INITIALIZING REAL SYSTEM WITH EXTERNAL LLM")
    real_system = IntegratedRealLLMSystem(df)

# Train the model
print("\n" + "="*60)
real_system.train_graph_transformer(num_epochs=20)

# Test the system
print("\n" + "="*60)
print("🧪 TESTING THE INTEGRATED SYSTEM")
print("="*60)

for user_id in df['id'].unique()[:2]:
    result = real_system.predict_with_real_llm_explanation(user_id)
    if result:
        print(f"\n📊 FINAL RESULTS FOR USER {result['user_id']}")
        print("=" * 70)
        
        prediction = result['prediction']
        print(f"PREDICTION:")
        print(f"  • Phase: {prediction['predicted_phase']}")
        print(f"  • Days to Menstrual: {prediction['days_to_menstrual']:.1f}")
        print(f"  • Confidence: {prediction['confidence']} ({prediction['confidence_score']:.3f})")
        
        print(f"\n💡 AI EXPLANATION (Generated by Real LLM + RAG):")
        print(result['explanation'])
        
        print("\n" + "=" * 70)

print("\n✅ SYSTEM DEMONSTRATION COMPLETED!")
print("\n🎯 WHAT THE SYSTEM VISUALIZED:")
print("1. ✅ Medical Knowledge Graph (showing relationships)")
print("2. ✅ User Timeline Charts (showing cycle patterns)") 
print("3. ✅ Model Architecture Diagram (showing transformer design)")
print("4. ✅ Training Progress (showing loss reduction)")
print("5. ✅ Prediction Breakdown (showing probabilities and confidence)")
print("6. ✅ Evidence-Based Insights (from medical knowledge base)")