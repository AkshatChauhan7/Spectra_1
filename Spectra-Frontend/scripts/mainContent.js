// // scripts/mainContent.js

// let spectraState = "STOPPED"; 
// let readingQueue = [];
// let currentIndex = 0;
// let lastAccessedImageIndex = -1;
// let recognition = null;

// console.log("🚀 Spectra Content Script Injected and Ready (Mac Optimized)!");

// function initVoiceRecognition() {
//     if (!('webkitSpeechRecognition' in window)) {
//         console.warn("⚠️ Speech recognition not supported.");
//         return;
//     }
    
//     recognition = new webkitSpeechRecognition();
//     recognition.continuous = true;
//     recognition.interimResults = true; // Lightning-fast mode
//     recognition.lang = 'en-US';

//     recognition.onresult = (event) => {
//         for (let i = event.resultIndex; i < event.results.length; ++i) {
//             const transcript = event.results[i][0].transcript.trim().toLowerCase();
            
//             // ⚡ FAST INTERIM COMMANDS: Instantly react without waiting for silence
//             if (!event.results[i].isFinal) {
//                 if (transcript.includes("pause") || transcript.includes("stop")) {
//                     console.log(`⚡ Voice Heard: "${transcript}" -> Pausing`);
//                     handleCommand("pause", "voice");
//                     recognition.abort(); return; 
//                 }
//                 if (transcript.includes("next")) {
//                     console.log(`⚡ Voice Heard: "${transcript}" -> Skipping to Next Chart`);
//                     handleCommand("next", "voice");
//                     recognition.abort(); return; 
//                 }
//                 if (transcript.includes("previous") || transcript.includes("back")) {
//                     console.log(`⚡ Voice Heard: "${transcript}" -> Going to Previous Chart`);
//                     handleCommand("previous", "voice");
//                     recognition.abort(); return; 
//                 }
//                 if (transcript.includes("resume") || transcript.includes("play")) {
//                     console.log(`⚡ Voice Heard: "${transcript}" -> Resuming`);
//                     handleCommand("resume", "voice");
//                     recognition.abort(); return; 
//                 }
//             } else {
//                 // 🗣️ FINAL COMMANDS: Wait for the full sentence for "Ask" queries
//                 console.log(`🗣️ Finalized Voice Command: "${transcript}"`);
//                 if (transcript.startsWith("ask ") || transcript.startsWith("question ")) {
//                     handleCommand(transcript, "voice");
//                 }
//             }
//         }
//     };

//     recognition.onerror = (event) => {
//         if (event.error !== "aborted") console.error("🎤 Voice Error:", event.error);
//     };

//     recognition.onend = () => {
//         if (spectraState !== "STOPPED") {
//             try { recognition.start(); } catch(e) {}
//         }
//     };
// }

// function handleCommand(cmd, source = "keyboard") {
//     console.log(`⚙️ Command: '${cmd}' | State: ${spectraState}`);
    
//     if (spectraState === "STOPPED" && !cmd.includes("start")) return;

//     if (cmd.includes("stop") || (cmd === "end")) {
//         stopSpectra();
//     } 
//     else if (cmd.includes("pause")) {
//         spectraState = "PAUSED";
//         // Mac Fix: Use STOP instead of PAUSE to guarantee it shuts up immediately
//         chrome.runtime.sendMessage({ action: "stop" }); 
//     } 
//     else if (cmd.includes("resume") || cmd.includes("play")) {
//         if (spectraState === "PAUSED") {
//             spectraState = "READING";
//             processCurrentItem(); // Mac Fix: Simply restart reading the current item
//         }
//     } 
//     else if (cmd.includes("next")) {
//         jumpToImage(1);
//     } 
//     else if (cmd.includes("previous") || cmd.includes("back")) {
//         jumpToImage(-1);
//     } 
//     else if (cmd.startsWith("ask ") || cmd.startsWith("question") || source === "keyboard") {
//         spectraState = "PAUSED";
//         chrome.runtime.sendMessage({ action: "stop" }); // Stop reading background text
        
//         if (source === "keyboard" && cmd === "ask") {
//             const query = prompt("What is your question about the last chart?");
//             if (query) askQueryAboutLastGraph(query);
//         } else {
//             let query = cmd.replace("ask", "").replace("question", "").trim();
//             if (query) askQueryAboutLastGraph(query);
//         }
//     }
// }

// async function processCurrentItem() {
//     if (spectraState !== "READING") return;

//     if (currentIndex >= readingQueue.length) {
//         speak("I have finished reading the page.", false);
//         stopSpectra();
//         return;
//     }

//     const item = readingQueue[currentIndex];

//     if (item.type === 'text') {
//         speak(item.content, true); 
//     } 
//     else if (item.type === 'image') {
//         lastAccessedImageIndex = currentIndex;
//         speak("Encountered a chart. Analyzing...", false);
        
//         console.log("🌐 Sending Image to FastAPI Backend:", item.url);
//         const summary = await api.analyzeChart(item.url);
//         console.log("✅ Received Summary from Backend:", summary);
        
//         if (spectraState !== "READING") return; 
//         speak(summary, true);
//     }
// }

// function speak(text, notifyEnd) {
//     chrome.runtime.sendMessage({ action: "speak", text: text, notifyEnd: notifyEnd });
// }

// function toggleSpectra() {
//     if (spectraState === "STOPPED") {
//         startSpectra();
//     } else {
//         stopSpectra();
//     }
// }

// function startSpectra() {
//     console.log("🟢 Starting Spectra...");
//     readingQueue = domExtractor.buildReadingQueue();
//     if (readingQueue.length === 0) {
//         speak("I could not find any readable content on this page.", false);
//         return;
//     }

//     spectraState = "READING";
//     currentIndex = 0;
//     lastAccessedImageIndex = -1;

//     if (!recognition) initVoiceRecognition();
//     try { recognition.start(); } catch(e) {} 

//     speak("Spectra started. Reading page.", false);
//     setTimeout(processCurrentItem, 2500); 
// }

// function stopSpectra() {
//     console.log("🔴 Stopping Spectra...");
//     spectraState = "STOPPED";
//     chrome.runtime.sendMessage({ action: "stop" });
//     try { recognition.stop(); } catch(e) {}
// }

// function jumpToImage(direction) {
//     spectraState = "READING";
//     chrome.runtime.sendMessage({ action: "stop" }); 

//     let found = false;
//     let tempIndex = currentIndex + direction;

//     while (tempIndex >= 0 && tempIndex < readingQueue.length) {
//         if (readingQueue[tempIndex].type === 'image') {
//             currentIndex = tempIndex;
//             lastAccessedImageIndex = tempIndex;
//             found = true;
//             break;
//         }
//         tempIndex += direction;
//     }

//     if (found) {
//         processCurrentItem();
//     } else {
//         speak("There are no more charts in that direction.", false);
//         spectraState = "PAUSED"; 
//     }
// }

// async function askQueryAboutLastGraph(query) {
//     if (lastAccessedImageIndex === -1) {
//         speak("No chart has been analyzed yet.", false);
//         return;
//     }

//     speak("Processing your question...", false);
//     const targetUrl = readingQueue[lastAccessedImageIndex].url;
//     const answer = await api.askChartQuestion(targetUrl, query);
    
//     speak(answer, false); 
//     // It remains paused. The user must say "Resume" or press Option+R to continue reading the page.
// }

// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//     if (request.action === "toggle_spectra") {
//         toggleSpectra();
//     }
//     else if (request.action === "speech_ended" && spectraState === "READING") {
//         currentIndex++;
//         processCurrentItem();
//     }
// });

// // MAC KEYBOARD FIX: Use event.code instead of event.key
// document.addEventListener('keydown', (event) => {
//     if (!event.altKey) return; // Note: event.altKey detects the Option key on Mac

//     if (event.code === 'KeyP') handleCommand("pause", "keyboard");
//     if (event.code === 'KeyR') handleCommand("resume", "keyboard");
//     if (event.code === 'KeyN') handleCommand("next", "keyboard");
//     if (event.code === 'KeyB') handleCommand("previous", "keyboard");
//     if (event.code === 'KeyQ') handleCommand("ask", "keyboard");
// });



// scripts/mainContent.js

let spectraState = "STOPPED"; 
let readingQueue = [];
let currentIndex = 0;
let lastAccessedImageIndex = -1;
let recognition = null;
let isListeningForQuestion = false; // 🧠 NEW: Gives Spectra short-term memory!

console.log("🚀 Spectra Content Script Injected and Ready (Mac Optimized + Split Voice Fix)!");

function initVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
        console.warn("⚠️ Speech recognition not supported.");
        return;
    }
    
    recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true; 
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript.trim().toLowerCase();
            
            // ⚡ FAST INTERIM COMMANDS
            if (!event.results[i].isFinal) {
                if (transcript.includes("pause") || transcript.includes("stop")) {
                    handleCommand("pause", "voice");
                    recognition.abort(); return; 
                }
                if (transcript.includes("next")) {
                    handleCommand("next", "voice");
                    recognition.abort(); return; 
                }
                if (transcript.includes("previous") || transcript.includes("back")) {
                    handleCommand("previous", "voice");
                    recognition.abort(); return; 
                }
                if (transcript.includes("resume") || transcript.includes("play")) {
                    handleCommand("resume", "voice");
                    recognition.abort(); return; 
                }
            } else {
                // 🗣️ FINAL COMMANDS
                console.log(`🗣️ Finalized Voice Command: "${transcript}"`);
                
                // 1. If it heard "ask" a second ago, treat THIS as the question
                if (isListeningForQuestion) {
                    isListeningForQuestion = false;
                    handleCommand(`ask ${transcript}`, "voice");
                    return;
                }

                // 2. If it ONLY heard "ask", wait for the next sentence!
                if (transcript === "ask" || transcript === "question") {
                    console.log("👂 Waiting for the rest of the question...");
                    isListeningForQuestion = true;
                    return;
                }

                // 3. Normal all-in-one sentence (e.g., "ask what is the value")
                if (transcript.startsWith("ask ") || transcript.startsWith("question ")) {
                    handleCommand(transcript, "voice");
                }
            }
        }
    };

    recognition.onerror = (event) => {
        // Suppress harmless "aborted" and "no-speech" errors
        if (event.error !== "aborted" && event.error !== "no-speech") {
            console.error("🎤 Voice Error:", event.error);
        }
    };

    recognition.onend = () => {
        if (spectraState !== "STOPPED") {
            try { recognition.start(); } catch(e) {}
        }
    };
}

function handleCommand(cmd, source = "keyboard") {
    console.log(`⚙️ Command: '${cmd}' | State: ${spectraState}`);
    
    if (spectraState === "STOPPED" && !cmd.includes("start")) return;

    if (cmd.includes("stop") || (cmd === "end")) {
        stopSpectra();
    } 
    else if (cmd.includes("pause")) {
        spectraState = "PAUSED";
        chrome.runtime.sendMessage({ action: "stop" }); 
    } 
    else if (cmd.includes("resume") || cmd.includes("play")) {
        if (spectraState === "PAUSED") {
            spectraState = "READING";
            processCurrentItem(); 
        }
    } 
    else if (cmd.includes("next")) {
        jumpToImage(1);
    } 
    else if (cmd.includes("previous") || cmd.includes("back")) {
        jumpToImage(-1);
    } 
    else if (cmd.startsWith("ask ") || cmd.startsWith("question") || source === "keyboard") {
        spectraState = "PAUSED";
        chrome.runtime.sendMessage({ action: "stop" }); 
        
        if (source === "keyboard" && cmd === "ask") {
            const query = prompt("What is your question about the last chart?");
            if (query) askQueryAboutLastGraph(query);
        } else {
            let query = cmd.replace("ask", "").replace("question", "").trim();
            if (query) askQueryAboutLastGraph(query);
        }
    }
}

async function processCurrentItem() {
    if (spectraState !== "READING") return;

    if (currentIndex >= readingQueue.length) {
        speak("I have finished reading the page.", false);
        stopSpectra();
        return;
    }

    const item = readingQueue[currentIndex];

    if (item.type === 'text') {
        speak(item.content, true); 
    } 
    else if (item.type === 'image') {
        lastAccessedImageIndex = currentIndex;
        speak("Encountered a chart. Analyzing...", false);
        
        console.log("🌐 Sending Image to FastAPI Backend:", item.url);
        const summary = await api.analyzeChart(item.url);
        console.log("✅ Received Summary from Backend:", summary);
        
        if (spectraState !== "READING") return; 
        speak(summary, true);
    }
}

function speak(text, notifyEnd) {
    chrome.runtime.sendMessage({ action: "speak", text: text, notifyEnd: notifyEnd });
}

function toggleSpectra() {
    if (spectraState === "STOPPED") {
        startSpectra();
    } else {
        stopSpectra();
    }
}

function startSpectra() {
    console.log("🟢 Starting Spectra...");
    readingQueue = domExtractor.buildReadingQueue();
    if (readingQueue.length === 0) {
        speak("I could not find any readable content on this page.", false);
        return;
    }

    spectraState = "READING";
    currentIndex = 0;
    lastAccessedImageIndex = -1;
    isListeningForQuestion = false; // Reset memory on start

    if (!recognition) initVoiceRecognition();
    try { recognition.start(); } catch(e) {} 

    speak("Spectra started. Reading page.", false);
    setTimeout(processCurrentItem, 2500); 
}

function stopSpectra() {
    console.log("🔴 Stopping Spectra...");
    spectraState = "STOPPED";
    chrome.runtime.sendMessage({ action: "stop" });
    try { recognition.stop(); } catch(e) {}
}

function jumpToImage(direction) {
    spectraState = "READING";
    chrome.runtime.sendMessage({ action: "stop" }); 

    let found = false;
    let tempIndex = currentIndex + direction;

    while (tempIndex >= 0 && tempIndex < readingQueue.length) {
        if (readingQueue[tempIndex].type === 'image') {
            currentIndex = tempIndex;
            lastAccessedImageIndex = tempIndex;
            found = true;
            break;
        }
        tempIndex += direction;
    }

    if (found) {
        processCurrentItem();
    } else {
        speak("There are no more charts in that direction.", false);
        spectraState = "PAUSED"; 
    }
}

async function askQueryAboutLastGraph(query) {
    if (lastAccessedImageIndex === -1) {
        speak("No chart has been analyzed yet.", false);
        return;
    }

    speak("Processing your question...", false);
    const targetUrl = readingQueue[lastAccessedImageIndex].url;
    const answer = await api.askChartQuestion(targetUrl, query);
    
    speak(answer, false); 
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggle_spectra") {
        toggleSpectra();
    }
    else if (request.action === "speech_ended" && spectraState === "READING") {
        currentIndex++;
        processCurrentItem();
    }
});

document.addEventListener('keydown', (event) => {
    if (!event.altKey) return; 

    if (event.code === 'KeyP') handleCommand("pause", "keyboard");
    if (event.code === 'KeyR') handleCommand("resume", "keyboard");
    if (event.code === 'KeyN') handleCommand("next", "keyboard");
    if (event.code === 'KeyB') handleCommand("previous", "keyboard");
    if (event.code === 'KeyQ') handleCommand("ask", "keyboard");
});