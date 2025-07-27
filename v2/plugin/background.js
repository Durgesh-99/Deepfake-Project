// Context menu: Right-click on image to check for deepfake
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "deepfake-check",
        title: "Check for Deepfake",
        contexts: ["image"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "deepfake-check" && info.srcUrl) {
        fetch(info.srcUrl)
            .then(response => response.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append("image", blob, "image.jpg");
                return fetch("http://127.0.0.1:8000/", { method: "POST", body: formData });
            })
            .then(response => response.json())
            .then(data => {
                let resultMessage = data.class === 0 ? "🚨 Image is Fake" : "✅ Image is Real";
                resultMessage += `\nConfidence: ${(data.confidence * 100).toFixed(2)}%`;
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    func: msg => alert(msg),
                    args: [resultMessage]
                });
            })
            .catch(error => {
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    func: msg => alert(msg),
                    args: [`Error checking image: ${error.message}`]
                });
            });
    }
});
