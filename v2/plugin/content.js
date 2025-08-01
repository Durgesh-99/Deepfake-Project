let abortController = new AbortController();
let toggle = false;

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
    if(!toggle) return;
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
    abortController.abort();                  // Stop all active fetches
    abortController = new AbortController();
    document.querySelectorAll("img").forEach(image => {
        image.style.border = ""; // Remove border when toggled OFF
    });
}

// On load and on detection toggle
async function initialize() {
    const isEnabled = await getDetectionStatus();
    toggle = isEnabled;
    if (isEnabled) checkImages();
}

chrome.storage.onChanged.addListener(async (changes, namespace) => {
    if (changes.detectionEnabled) {
        toggle = changes.detectionEnabled.newValue;
        if (toggle) {
            checkImages();
        } else {
            stopProcessing();
        }
    }
});

initialize();