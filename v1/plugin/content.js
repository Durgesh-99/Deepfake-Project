let abortController = new AbortController();

function addOverlay(image, isReal) {
    const existingOverlay = image.parentElement.querySelector(".deepfake-overlay");
    if (existingOverlay) existingOverlay.remove();

    image.style.border = isReal ? "3px solid green" : "3px solid red";
}

// ✅ Convert chrome.storage.sync.get() into a Promise-based function
function getDetectionStatus() {
    return new Promise((resolve) => {
        chrome.storage.sync.get("detectionEnabled", (data) => {
            resolve(data.detectionEnabled || false); // Default to false if undefined
        });
    });
}

async function processImage(image) {
    try {
        const response = await fetch(image.src, { signal: abortController.signal });
        const blob = await response.blob();
        const formData = new FormData();
        formData.append("image", blob, "image.jpg");

        const serverResponse = await fetch("http://localhost:3000/", {
            method: "POST",
            body: formData,
            signal: abortController.signal // Attach abort signal
        });

        const data = await serverResponse.json();
        addOverlay(image, data.class === 1);
    } catch (error) { }
}

function stopProcessing() {
    abortController.abort(); // Stops all fetch requests
    abortController = new AbortController(); // Reset for future requests

    // Remove all overlays
    document.querySelectorAll("img").forEach(image => {
        image.style.border = ""; // Remove border when toggled OFF
    });
}

async function checkImages() {
    const isEnabled = await getDetectionStatus();
    if (!isEnabled) return;

    const images = Array.from(document.querySelectorAll("img"));
    const batchSize = 20;

    for (let i = 0; i < images.length; i += batchSize) {
        const batch = images.slice(i, i + batchSize);
        await Promise.all(batch.map(image => processImage(image)));

        // Check if the detection was disabled in the middle
        const stillEnabled = await getDetectionStatus();
        if (!stillEnabled) {
            stopProcessing();
            return;
        }
    }
}

// ✅ Wait for storage retrieval before calling checkImages()
async function initialize() {
    const isEnabled = await getDetectionStatus();
    if (isEnabled) {
        checkImages();
    }
}

// ✅ Run only after we confirm the storage value
initialize();

chrome.storage.onChanged.addListener(async (changes, namespace) => {
    if (changes.detectionEnabled) {
        if (changes.detectionEnabled.newValue) {
            checkImages(); // ✅ Start detection when toggled ON
        } else {
            stopProcessing(); // ✅ Stop processing when toggled OFF
        }
    }
});
