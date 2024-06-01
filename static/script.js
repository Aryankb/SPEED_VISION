document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('videoPlayer');
    const canvas = document.getElementById('videoCanvas');
    const ctx = canvas.getContext('2d');
    let drawingLine = false;
    let drawingFirstLine = true;
    let drawingSecondLine = false;
    let firstLineStart = { x: 0, y: 0 };
    let firstLineEnd = { x: 0, y: 0 };
    let secondLineStart = { x: 0, y: 0 };
    let secondLineEnd = { x: 0, y: 0 };

    // Load and play the video
    const videoPath = document.getElementById('videoPath').value;
    video.src = videoPath;
    video.autoplay = true;

    // Set canvas dimensions to match the video
    video.addEventListener('loadedmetadata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        draw();
    });

    // Event listeners for mouse down and up
    canvas.addEventListener('mousedown', (event) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;

        if (drawingFirstLine) {
            firstLineStart = { x, y };
            drawingLine = true;
        } else if (drawingSecondLine) {
            secondLineStart = { x, y };
            drawingLine = true;
        }
    });

    canvas.addEventListener('mouseup', (event) => {
        if (drawingLine) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (event.clientX - rect.left) * scaleX;
            const y = (event.clientY - rect.top) * scaleY;

            if (drawingFirstLine) {
                firstLineEnd = { x, y };
                drawingLine = false;
                drawingFirstLine = false;
                drawingSecondLine = true;
                alert('First line drawn. Now, draw the second line. Click to set the start point.');
            } else if (drawingSecondLine) {
                secondLineEnd = { x, y };
                drawingLine = false;
                drawingSecondLine = false;
                // Enable the detect button after drawing the second line
                document.getElementById('detectButton').disabled = false;
            }
        }
    });

    // Function to draw lines
    function drawLine(start, end, color) {
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = color; // Set line color
        ctx.lineWidth = 2; // Set line width
        ctx.stroke();
    }

    // Continuous drawing function to ensure both lines stay visible
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (firstLineEnd.x !== 0 || firstLineEnd.y !== 0) {
            drawLine(firstLineStart, firstLineEnd, 'red');
        }
        if (secondLineEnd.x !== 0 || secondLineEnd.y !== 0) {
            drawLine(secondLineStart, secondLineEnd, 'blue');
        }
        requestAnimationFrame(draw);
    }

    // Event listener for overspeeding button
    document.getElementById('detectButton').addEventListener('click', () => {
        // Logic for detecting overspeeding vehicles
        alert('Detecting overspeeding vehicles...');
    });
});
