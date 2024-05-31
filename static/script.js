document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('videoPlayer');
    const canvas = document.getElementById('videoCanvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let start = { x: 0, y: 0 };
    let end = { x: 0, y: 0 };

    // Load and play the video
    const videoPath = document.getElementById('videoPath').value;
    video.src = videoPath;
    video.autoplay = true;

    // Set canvas dimensions to match the video
    video.addEventListener('loadedmetadata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    });

    // Event listeners for mouse down, move, and up
    canvas.addEventListener('mousedown', (event) => {
        start.x = event.offsetX;
        start.y = event.offsetY;
        drawing = true;
    });

    canvas.addEventListener('mousemove', (event) => {
        if (drawing) {
            end.x = event.offsetX;
            end.y = event.offsetY;
            drawLine(start, end);
            start.x = end.x;
            start.y = end.y;
        }
    });

    canvas.addEventListener('mouseup', () => {
        drawing = false;
    });

    // Function to draw lines
    function drawLine(start, end) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = 'red'; // Change line color to red
        ctx.lineWidth = 2; // Set line width
        ctx.stroke();
    }

    // Event listener for overspeeding button
    document.getElementById('detectButton').addEventListener('click', () => {
        // Logic for detecting overspeeding vehicles
    });
});
