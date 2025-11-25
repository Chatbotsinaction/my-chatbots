# Web Interface for Chatbot using Flask
from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime

# Import our chatbot class (assuming it's in a separate file)
from chatbot3 import MLChatbot

app = Flask(__name__)

# Initialize chatbot
bot = MLChatbot()
print(bot.get_best_response(""))

# Store conversation sessions (in production, use a database)
conversations = {}

@app.route('/')
def index():
    """
    Serve the main chat interface
    """
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """
    Handle chat requests via API
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please type a message to chat with me!'
            }), 400

        # ----------------------------
        # Get response from chatbot
        # ----------------------------
        response_data = bot.get_best_response(user_message)

        # Ensure it is a tuple (response, confidence)
        if not isinstance(response_data, tuple) or len(response_data) != 2:
            response_data = (str(response_data), None)

        response, confidence = response_data

        # If response is None or empty, fallback
        if not response:
            response = bot.generate_fallback_response(user_message)

        bot_response = str(response)

        # Debugging prints (optional)
        print("DEBUG user message:", user_message)
        print("DEBUG bot response:", bot_response)
        print("DEBUG confidence:", confidence)

        # ----------------------------
        # Store conversation
        # ----------------------------
        if session_id not in conversations:
            conversations[session_id] = []

        conversations[session_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })

        # Return JSON response
        return jsonify({
            'response': bot_response,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print("ERROR:", e)  # Print exception in terminal
        return jsonify({
            'error': str(e),
            'response': 'Sorry, I encountered an error. Please try again.'
        }), 500

@app.route('/api/history/<session_id>')
def get_history(session_id):
    """
    Get conversation history for a session
    """
    history = conversations.get(session_id, [])
    return jsonify({'history': history})

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Production server
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)