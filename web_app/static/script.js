document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('capture-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const resultImage = document.getElementById('result-image');
    const loader = document.getElementById('loader');

    // Stats
    const countTotal = document.getElementById('count-total');
    const countFresh = document.getElementById('count-fresh');
    const countRotten = document.getElementById('count-rotten');

    let currentFile = null;
    let currentImageBase64 = null;
    let stream = null;

    // Tab Switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');

            if (tab.dataset.target === 'camera-tab') {
                startCamera();
            } else {
                stopCamera();
            }
        });
    });

    // File Upload Handling
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            currentFile = e.target.files[0];
            currentImageBase64 = null; // Reset camera image
            analyzeBtn.disabled = false;

            // Show preview (optional, but good for UX)
            const reader = new FileReader();
            reader.onload = (e) => {
                // You could show a preview here if you wanted
                console.log("File loaded");
            };
            reader.readAsDataURL(currentFile);

            dropZone.querySelector('p').textContent = currentFile.name;
        }
    });

    // Camera Handling
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Impossible d'accéder à la caméra.");
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        currentImageBase64 = canvas.toDataURL('image/jpeg');
        currentFile = null; // Reset file

        analyzeBtn.disabled = false;

        // Visual feedback
        video.style.opacity = "0.5";
        setTimeout(() => video.style.opacity = "1", 200);
    });

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();

        if (currentFile) {
            formData.append('file', currentFile);
        } else if (currentImageBase64) {
            formData.append('image', currentImageBase64);
        } else {
            alert("Aucune image sélectionnée !");
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert("Erreur: " + data.error);
            } else {
                // Update UI
                resultImage.src = data.image_url;
                countTotal.textContent = data.total;
                countFresh.textContent = data.fresh;
                countRotten.textContent = data.rotten;

                // Scroll to result
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (err) {
            console.error(err);
            alert("Erreur lors de l'analyse.");
        } finally {
            loader.classList.add('hidden');
        }
    });
});
