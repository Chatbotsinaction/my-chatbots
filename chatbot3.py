# Machine Learning Chatbot with Scikit-learn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class MLChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Training data - in a real application, this would come from a larger dataset
        self.training_data = pd.DataFrame({
            'questions': [
                "Who is Sasha Boshno?",
                "What is Python programming?",
                "How do I install Python?",
                "What are Python libraries?",
                "How to create a function in Python?",
                "What is machine learning?",
                "How do chatbots work?",
                "What is artificial intelligence?",
                "How to learn programming?",
                "What is natural language processing?",
                "How to build a website?",
                "What are APIs?",
                "How do databases work?",
                "What is web scraping?",
                "How to handle errors in code?",
                "What are data structures?",
                "How to optimize code performance?",
                "What is version control?",
                "How to test software?",
                "What are design patterns?",
                "How to deploy applications?",
                "What is cloud computing?",
                "How do neural networks work?",
                "What is deep learning?",
                "How to analyze data?",
                "What are algorithms?",
            ],
            'answers': [
                "A really smart guy.",
                "Python is a high-level, interpreted programming language known for its simplicity and readability. It's excellent for beginners and powerful enough for complex applications.",
                "You can install Python by downloading it from python.org. Most systems also allow installation via package managers like apt, brew, or chocolatey.",
                "Python libraries are pre-written code modules that extend Python's functionality. Popular ones include NumPy for math, Pandas for data analysis, and Requests for web APIs.",
                "Use the 'def' keyword followed by the function name and parameters. For example: def greet(name): return f'Hello, {name}!'",
                "Machine learning is a subset of AI that enables computers to learn patterns from data without being explicitly programmed for every scenario.",
                "Chatbots work by processing user input, understanding intent, and generating appropriate responses using various techniques from simple rules to advanced AI.",
                "Artificial Intelligence is the simulation of human intelligence in machines, enabling them to think, learn, and make decisions.",
                "Start with fundamentals, practice regularly, build projects, and join communities. Choose a language like Python and stick with it initially.",
                "NLP is a field of AI that helps computers understand, interpret, and generate human language in a valuable way.",
                "You can build websites using HTML/CSS for structure and styling, JavaScript for interactivity, and frameworks like React or Django for complex applications.",
                "APIs (Application Programming Interfaces) are sets of protocols that allow different software applications to communicate with each other.",
                "Databases store and organize data systematically, allowing efficient retrieval, updating, and management through query languages like SQL.",
                "Web scraping is the process of automatically extracting data from websites using tools like Beautiful Soup or Scrapy in Python.",
                "Handle errors using try-except blocks, validate input data, write defensive code, and implement proper logging for debugging.",
                "Data structures organize and store data efficiently. Common ones include lists, dictionaries, stacks, queues, and trees.",
                "Optimize code by choosing efficient algorithms, avoiding unnecessary operations, using appropriate data structures, and profiling performance.",
                "Version control systems like Git track changes in code over time, enabling collaboration and maintaining project history.",
                "Software testing involves writing automated tests, manual testing, and using frameworks like pytest to ensure code quality and reliability.",
                "Design patterns are reusable solutions to common programming problems, helping create maintainable and scalable software architectures.",
                "Deploy applications using cloud platforms like AWS, Heroku, or DigitalOcean, containerization with Docker, or traditional servers.",
                "Cloud computing delivers computing services over the internet, offering scalable resources without managing physical hardware.",
                "Neural networks are AI models inspired by biological neurons, learning patterns through interconnected layers of nodes.",
                "Deep learning uses multi-layered neural networks to automatically learn complex patterns from large amounts of data.",
                "Data analysis involves examining datasets to discover patterns, trends, and insights using tools like Python, R, or specialized software.",
                "Algorithms are step-by-step procedures for solving problems or performing tasks, fundamental to all computer programming."
            ]
        })
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)  # Include both single words and pairs
        )
        
        # Fit the vectorizer on training questions
        self.question_vectors = self.vectorizer.fit_transform(self.training_data['questions'])
        
        # Conversation history for context
        self.conversation_history = []
    
    def preprocess_text(self, text):
        """
        Clean and normalize text input
        """
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and lemmatize
        tokens = text.split()
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def get_best_response(self, user_input, confidence_threshold=0.1):
        """
        Find the most similar question and return corresponding answer
        """
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize the user input
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate similarities with all training questions
        similarities = cosine_similarity(user_vector, self.question_vectors)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        confidence = similarities[0][best_match_idx]
        
        # Return response if confidence is above threshold
        if confidence > confidence_threshold:
            response = self.training_data.iloc[best_match_idx]['answers']
            return response, confidence
        else:
            return None, confidence
    
    def generate_fallback_response(self, user_input):
        """
        Generate contextual fallback responses when no good match is found
        """
        fallback_responses = [
            "That's an interesting question! I'm still learning about that topic. Could you ask it in a different way?",
            "I don't have specific information about that right now, but I'd love to learn more! What else would you like to know?",
            "That's outside my current knowledge base, but it sounds fascinating! Can you tell me more about your interest in this topic?",
            "I'm not quite sure about that yet. My training focused more on programming and technology topics. What else can I help you with?",
            "That's a great question that I need to learn more about! Is there something related I might be able to help with?"
        ]
        return random.choice(fallback_responses)
    
    def chat(self):
        """
        Start interactive ML-powered conversation
        """
        print("🤖 ML Chatbot: Hello! I'm a machine learning chatbot trained on programming and technology topics.")
        print("I use similarity matching to find the best responses to your questions.")
        print("Ask me about Python, programming, AI, or technology in general!")
        print("Type 'quit' to exit.")
        print("-" * 70)
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'goodbye']:
                print("🤖 ML Chatbot: Thanks for chatting! Keep learning and coding!")
                break
            
            if not user_input.strip():
                continue
            
            # Get response from ML model
            response, confidence = self.get_best_response(user_input)
            
            if response:
                print(f"🤖 ML Chatbot: {response}")
                print(f"   (Confidence: {confidence:.2f})")
            else:
                fallback = self.generate_fallback_response(user_input)
                print(f"🤖 ML Chatbot: {fallback}")
                print(f"   (Low confidence: {confidence:.2f})")
            
            # Store conversation for potential future context
            self.conversation_history.append({
                'user': user_input,
                'bot': response if response else fallback,
                'confidence': confidence
            })

# Create and run the ML chatbot
if __name__ == "__main__":
    bot = MLChatbot()
    bot.chat()