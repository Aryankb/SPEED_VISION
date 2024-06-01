import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'videoFile' not in request.files:
        return "No video file uploaded", 400

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return "No selected file", 400

    if 'videoName' not in request.form:
        return "Video name not provided", 400

    video_name = request.form['videoName']

    if video_file:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        filename = video_file.filename
        video_path = os.path.join(folder_path, filename)
        video_file.save(video_path)
        
        return render_template('index.html', video_path=video_path)

    return "Video uploaded successfully!"

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    first_line_start = data['firstLineStart']
    first_line_end = data['firstLineEnd']
    second_line_start = data['secondLineStart']
    second_line_end = data['secondLineEnd']

    # Placeholder for actual detection logic
    print(f'First Line Start: {first_line_start}')
    print(f'First Line End: {first_line_end}')
    print(f'Second Line Start: {second_line_start}')
    print(f'Second Line End: {second_line_end}')
    
    # Example response
    result = {
        'status': 'success',
        'message': 'Detection complete',
        'data': {
            'firstLine': {
                'start': first_line_start,
                'end': first_line_end
            },
            'secondLine': {
                'start': second_line_start,
                'end': second_line_end
            }
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
