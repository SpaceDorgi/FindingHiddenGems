from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import requests


app = Flask(__name__, static_folder='')
CORS(app)

# Cache for recommendations data - load once and reuse
_recommendations_cache = None

def get_recommendations():
    """Load recommendations with caching to avoid reloading on every request."""
    global _recommendations_cache
    
    if _recommendations_cache is not None:
        print(f"Using cached recommendations ({len(_recommendations_cache)} items)")
        return _recommendations_cache
    
    print("Loading recommendations (first time)...")
    try:
        # Use user_top25_recs.csv instead of JSON
        import csv
        _recommendations_cache = []
        with open('user_recs_25_with_gem_score.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields if needed
                if 'predicted_rating' in row:
                    try:
                        row['predicted_rating'] = float(row['predicted_rating'])
                    except Exception:
                        pass
                if 'stars' in row:
                    try:
                        row['stars'] = float(row['stars'])
                    except Exception:
                        pass
                _recommendations_cache.append(row)
        print(f"Successfully loaded and cached {len(_recommendations_cache)} recommendations from CSV")
        return _recommendations_cache
    except Exception as e:
        print(f"Failed to load recommendations from CSV: {str(e)}")
        raise

@app.route('/api/all-businesses')
def api_all_businesses():
    """Return all businesses with their hidden gem scores and metrics."""
    try:
        # Use the CloudFront URL for the JSON file
        json_url = 'https://d1tb01eg2pkw2c.cloudfront.net/data/all_businesses.json'
        print(f"Loading all businesses from: {json_url}")
        
        try:
            response = requests.get(json_url, timeout=30)
            response.raise_for_status()
            businesses = response.json()
            print(f"Successfully loaded {len(businesses)} businesses")
            return jsonify(businesses)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {str(je)}")
            return jsonify({'error': f'JSON parse error: {str(je)}'}), 500

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve the JSON file for recommendations
@app.route('/api/recommendations')
def api_recommendations():
    """Return user recommendations with pagination to avoid Lambda 6MB limit."""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 1000))
        
        recommendations = get_recommendations()
        
        # Calculate pagination
        total = len(recommendations)
        start = (page - 1) * per_page
        end = start + per_page
        
        paginated_data = recommendations[start:end]
        
        response_data = {
            'data': paginated_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': (total + per_page - 1) // per_page
            }
        }
        
        print(f"Returning page {page} with {len(paginated_data)} items (total: {total})")
        return jsonify(response_data)
        
    except FileNotFoundError:
        print("Local file not found")
        return jsonify({'error': 'Recommendations file not found'}), 404
    except Exception as e:
        print(f"Error loading recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Optional: serve static frontend so everything runs from Flask (if you want)
@app.route('/')
def index():
    response = send_from_directory('.', 'recommendation_interface.html')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/recommendations')
def recommendations():
    response = send_from_directory('.', 'recommendation_interface.html')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

def lambda_handler(event, context):
    """AWS Lambda handler that processes API Gateway events."""
    path = event.get('rawPath') or event.get('path', '/')
    method = event.get('requestContext', {}).get('http', {}).get('method') or event.get('httpMethod', 'GET')
    
    try:
        # Create a WSGI environment from the Lambda event
        from werkzeug.wrappers import Request, Response
        from io import BytesIO
        
        # Build environ dict for WSGI
        environ = {
            'REQUEST_METHOD': method,
            'SCRIPT_NAME': '',
            'PATH_INFO': path,
            'QUERY_STRING': event.get('rawQueryString', ''),
            'SERVER_NAME': 'lambda',
            'SERVER_PORT': '443',
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': 'https',
            'wsgi.input': BytesIO(),
            'wsgi.errors': BytesIO(),
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
        }
        
        # Call the Flask app
        response = Response.from_app(app, environ)
        
        # Return Lambda-compatible response
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True)
        }
    except Exception as e:
        print(f"Lambda handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }

#if __name__ == '__main__':
    # For quick testing (not production)
#    app.run(host='0.0.0.0', port=8000, debug=True)