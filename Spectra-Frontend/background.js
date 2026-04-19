// background.js

console.log("⚙️ Spectra Background Service Worker is running!");

chrome.action.onClicked.addListener((tab) => {
    console.log("🖱️ Extension Icon Clicked! Waking up tab:", tab.id);
    chrome.tabs.sendMessage(tab.id, { action: "toggle_spectra" });
});

chrome.commands.onCommand.addListener((command) => {
    console.log("⌨️ Global Shortcut Pressed:", command);
    if (command === "_execute_action") {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, { action: "toggle_spectra" });
        });
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log(`📩 Received Action: ${request.action}`);

    if (request.action === "speak") {
        chrome.tts.stop(); 
        console.log(`🗣️ Speaking: "${request.text.substring(0, 50)}..."`);
        
        chrome.tts.speak(request.text, {
            rate: 1.0,
            onEvent: function(event) {
                if (event.type === 'error') {
                    console.error("❌ TTS Error:", event.errorMessage);
                }
                if (event.type === 'end' && request.notifyEnd && sender.tab) {
                    console.log("✅ Finished speaking. Telling webpage to move to next item.");
                    chrome.tabs.sendMessage(sender.tab.id, { action: "speech_ended" });
                }
            }
        });
    } 
    else if (request.action === "pause") {
        console.log("⏸️ Pausing speech.");
        chrome.tts.pause();
    } 
    else if (request.action === "resume") {
        console.log("▶️ Resuming speech.");
        chrome.tts.resume();
    } 
    else if (request.action === "stop") {
        console.log("⏹️ Stopping speech completely.");
        chrome.tts.stop();
    }
});

// // const API_URL = "https://shadowgard3n-spectra-backend.hf.space/analyze-graph";
// const API_URL = "http://127.0.0.1:8000/analyze-graph";


// // 1. Listen for the icon click
// chrome.action.onClicked.addListener((tab) => {
//   console.log("🔘 Extension icon clicked on URL:", tab.url);
  
//   chrome.scripting.executeScript({
//     target: { tabId: tab.id, allFrames: false },
//     files: ["content.js"]
//   })
//   .then(() => console.log("✅ content.js injected successfully!"))
//   .catch((err) => console.error("❌ Failed to inject content.js:", err));
// });

// // 2. Listen for image URLs from the webpage
// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   if (request.action === "analyzeImage") {
//     console.log("🖼️ Received image to analyze:", request.src);
    
//     processImage(request.src)
//       .then(explanation => sendResponse({ text: explanation }))
//       .catch(error => {
//         console.error("❌ Background Fetch Error:", error);
//         sendResponse({ text: "Sorry, I could not process this image." });
//       });
      
//     return true; // Keeps the message channel open while waiting for FastAPI
//   }
// });

// // 3. Send to Hugging Face
// async function processImage(imgSrc) {
//   const imageResponse = await fetch(imgSrc);
//   const imageBlob = await imageResponse.blob();
//   const formData = new FormData();
//   formData.append("file", imageBlob, "image.png");

//   const apiResponse = await fetch(API_URL, {
//     method: "POST",
//     body: formData
//   });

//   if (!apiResponse.ok) throw new Error(`API Error: ${apiResponse.status}`);
//   return await apiResponse.text();
// }