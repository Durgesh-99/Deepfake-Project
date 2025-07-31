let abortController = new AbortController();

function addOverlay(image, isReal) {
    const existing = image.parentElement.querySelector(".deepfake-overlay");
    if (existing) existing.remove();
    image.style.border = isReal ? "3px solid green" : "3px solid red";
}

function getDetectionStatus() {
    return new Promise(resolve => {
        chrome.storage.sync.get("detectionEnabled", data => {
            resolve(data.detectionEnabled || false);
        });
    });
}

async function processImage(image) {
    try {
        const response = await fetch(image.src, { signal: abortController.signal });
        const blob = await response.blob();
        const formData = new FormData();
        formData.append("image", blob, "image.jpg");

        const res = await fetch("http://127.0.0.1:8000/", {
            method: "POST",
            body: formData,
            signal: abortController.signal
        });

        const data = await res.json();
        addOverlay(image, data.class === 1);
    } catch (error) {
        if (error.name === "AbortError") {
            console.log("Image processing aborted.");
            return;
        }
        console.error("Error processing image:", error);
    }
}

async function checkImages() {
    const isEnabled = await getDetectionStatus();
    if (!isEnabled) return;
    const images = document.querySelectorAll("img");
    for (const image of images) {
        await processImage(image); // Serial; switch to batch for speed if desired
    }
}

function stopProcessing() {
    abortController.abort(); // Stops all fetch requests
    abortController = new AbortController(); // Reset for future requests

    // Remove all overlays
    document.querySelectorAll("img").forEach(image => {
        image.style.border = ""; // Remove border when toggled OFF
    });
}

// Batch processing for all images (optional, boosts speed for large pages)
async function checkImagesBatch() {
    const isEnabled = await getDetectionStatus();
    if (!isEnabled) return;
    const images = Array.from(document.querySelectorAll("img"));
    const formData = new FormData();
    for (let i = 0; i < images.length; i++) {
        try {
            const r = await fetch(images[i].src);
            const blob = await r.blob();
            formData.append("images", blob, `img${i}.jpg`);

            const stillEnabled = await getDetectionStatus();
            if (!stillEnabled) {
                stopProcessing();
                return;
            }
        } catch (e) {}
    }
    const res = await fetch("http://127.0.0.1:8000/predict-batch", { method: "POST", body: formData, signal: abortController.signal});
    const results = await res.json();
    results.forEach((pred, i) => {
        if (typeof pred.class !== "undefined") {
            addOverlay(images[i], pred.class === 1);
        }
    });
}

// On load and on detection toggle
async function initialize() {
    const isEnabled = await getDetectionStatus();
    if (isEnabled) checkImages();
}
initialize();

chrome.storage.onChanged.addListener(async (changes, namespace) => {
    if (changes.detectionEnabled) {
        if (changes.detectionEnabled.newValue) {
            checkImages(); // or checkImagesBatch();
        } else {
            stopProcessing();
        }
    }
});

