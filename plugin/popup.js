document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("toggleDetection");
    const statusText = document.getElementById("status");

    // Load the saved state
    chrome.storage.sync.get("detectionEnabled", (data) => {
        toggle.checked = data.detectionEnabled || false;
        statusText.textContent = toggle.checked ? "Detection is ON" : "Detection is OFF";
    });

    toggle.addEventListener("change", () => {
        const isEnabled = toggle.checked;
        chrome.storage.sync.set({ detectionEnabled: isEnabled });
        statusText.textContent = isEnabled ? "Detection is ON" : "Detection is OFF";
    });
});

