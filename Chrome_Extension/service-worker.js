// service-worker.js
console.log('Service worker initialized');

// 아이콘 클릭으로 패널 열기/닫기 동작 설정
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error('Error setting panel behavior:', error));

chrome.runtime.onMessage.addListener((message, sender) => {
    console.log('Message received:', message);

    if (message.action === "openPanel") {
        chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
            try {
                await chrome.sidePanel.open({ tabId: tabs[0].id });
                await new Promise(resolve => setTimeout(resolve, 500));

                chrome.runtime.sendMessage({
                    action: "analyzeText",
                    text: message.text,
                    url: message.url,
                    type: message.type
                });
            } catch (error) {
                console.error('Error:', error);
            }
        });
    }
});