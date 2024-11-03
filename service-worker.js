chrome.runtime.onInstalled.addListener(() => {
    console.log("Extension installed");
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "openPanel") {
        // 사이드 패널에 텍스트를 보내기 위한 메시지 전송
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0].id) {
                chrome.scripting.executeScript({
                    target: { tabId: tabs[0].id },
                    func: (text) => {
                        chrome.runtime.sendMessage({ action: "showText", text: text });
                    },
                    args: [request.text],
                });
            }
        });
        sendResponse({ status: "opened" });
    }
});
