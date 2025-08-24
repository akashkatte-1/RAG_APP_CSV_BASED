from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize RAG system
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system with API key validation"""
    global rag_system
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False
        
        rag_system = RAGSystem(gemini_api_key)
        logger.info("RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page for the RAG application"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the CSV file
        if not rag_system:
            if not initialize_rag_system():
                return jsonify({'error': 'RAG system not initialized. Check your GEMINI_API_KEY'}), 500
        
        success, message = rag_system.process_csv(filepath)
        
        if success:
            return jsonify({
                'message': f'Successfully processed {filename}',
                'details': message
            })
        else:
            return jsonify({'error': message}), 500
            
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries to the RAG system"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        if not rag_system:
            if not initialize_rag_system():
                return jsonify({'error': 'RAG system not initialized. Check your GEMINI_API_KEY'}), 500
        
        # Get response from RAG system
        response = rag_system.query(question)
        
        return jsonify({
            'question': question,
            'answer': response['answer'],
            'sources': response.get('sources', [])
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check system status"""
    try:
        if not rag_system:
            initialize_rag_system()
        
        status_info = {
            'rag_system_initialized': rag_system is not None,
            'gemini_api_configured': bool(os.getenv('GEMINI_API_KEY')),
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
        }
        
        if rag_system:
            status_info.update(rag_system.get_status())
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
def reset_database():
    """Reset the vector database"""
    try:
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        success, message = rag_system.reset_database()
        
        if success:
            return jsonify({'message': message})
        else:
            return jsonify({'error': message}), 500
            
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize RAG system on startup
    initialize_rag_system()
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
