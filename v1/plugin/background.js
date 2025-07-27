// ✅ 1. Handle Right-Click Feature
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "deepfake-check",
        title: "Check for Deepfake",
        contexts: ["image"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "deepfake-check" && info.srcUrl) {
        const imageUrl = info.srcUrl;

        fetch(imageUrl)
            .then(response => response.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append("image", blob, "image.jpg");

                return fetch("http://localhost:3000/", {
                    method: "POST",
                    body: formData
                });
            })
            .then(response => response.json())
            .then(data => {
                let resultMessage = data.class === 0 ? "🚨 Image is Fake" : "✅ Image is Real";
                resultMessage += `\nConfidence: ${(data.confidence * 100).toFixed(2)}%`;

                // ✅ Show result using Chrome scripting
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: (msg) => alert(msg),
                    args: [resultMessage]
                });
            })
            .catch(error => {
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: (msg) => alert(msg),
                    args: [`Error checking image: ${error.message}`]
                });
            });
    }
});


chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "deepfake-check" && info.srcUrl) {
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: checkSingleImage,
            args: [info.srcUrl]
        });
    }
});

function checkSingleImage(imageUrl) {
    fetch(imageUrl)
        .then(response => response.blob())
        .then(blob => {
            const formData = new FormData();
            formData.append("image", blob, "image.jpg");

            return fetch("http://localhost:3000/", {
                method: "POST",
                body: formData
            });
        })
        .then(response => response.json())
        .catch(error => {
            alert("Error checking image: " + error.message);
        });
}


