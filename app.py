import os
from flask import Flask, render_template, request, jsonify
import subprocess
app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

video_name=""
filename=""
@app.route('/upload', methods=['POST'])
def upload():
    global video_name
    global filename
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
    global video_name
    global filename
    
    data = request.get_json()
    first_line_start = data['firstLineStart']
    first_line_end = data['firstLineEnd']
    second_line_start = data['secondLineStart']
    second_line_end = data['secondLineEnd']
    threshold_speed = data['thresholdSpeed']
    distance = data['distance']

    # Placeholder for actual detection logic
    print(f'Video Name: {video_name}')
    print(f'First Line Start: {first_line_start}')
    print(f'First Line End: {first_line_end}')
    print(f'Second Line Start: {second_line_start}')
    print(f'Second Line End: {second_line_end}')
    print(f'Threshold Speed: {threshold_speed}')
    print(f'Distance: {distance}')

    line_coordss = [
        first_line_start["x"], first_line_start["y"], first_line_end["x"], first_line_end["y"],
        second_line_start["x"], second_line_start["y"], second_line_end["x"], second_line_end["y"]
    ]

    line_coordss = map(int,line_coordss)
    if not (video_name and filename and threshold_speed and distance and line_coordss):
        return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400

    # Convert list of coordinates to a string
    line_coords_str = ' '.join(map(str, line_coordss))

    # Construct the command
    command = f'source /home/aryan/miniconda3/etc/profile.d/conda.sh && conda activate speed && python detect.py --conf 0.2 --device 0 --weights runs/train/exp10/weights/best.pt --project static/"{video_name}"/"detected_{filename}" --source static/"{video_name}"/"{filename}" --linecoordss {line_coords_str} --threshold {threshold_speed} --DISTAN {distance}'

    print(f'Running command: {command}')

    try:
        process = subprocess.run(command, shell=True, capture_output=True, text=True, executable='/bin/bash')
        print(f'Process return code: {process.returncode}')
        print(f'Process stdout: {process.stdout}')
        print(f'Process stderr: {process.stderr}')
        if process.returncode == 0:
            return jsonify({'status': 'success', 'output': process.stdout})
        else:
            return jsonify({'status': 'error', 'message': process.stderr}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
