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
  
          fetch("http://localhost:5000/", {
            method: "POST",
            body: formData
          })
          .then(response => {
            if (!response.ok) {
              throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
          })
          .then(data => {
            const message = `Image is ${data.class === 1 ? 'Real' : 'Fake'}\nConfidence: ${(data.confidence * 100).toFixed(2)}%`;
  
            chrome.scripting.executeScript({  // Inject content script to display alert
              target: { tabId: tab.id },
              function: (msg) => {
                alert(msg);
              },
              args: [message]
            });
  
          })
          .catch(error => {
            chrome.scripting.executeScript({ // Inject content script for error alert
              target: { tabId: tab.id },
              function: (msg) => {
                alert(msg);
              },
              args: [`Error checking image: ${error.message}`]
            });
          });
        })
        .catch(error => {
          chrome.scripting.executeScript({ // Inject content script for error alert
            target: { tabId: tab.id },
            function: (msg) => {
              alert(msg);
            },
            args: [`Error fetching image: ${error.message}`]
          });
        });
    }
  });