chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "showText") {
        const contentDiv = document.getElementById('content');
        contentDiv.innerText = request.text; // 전달받은 텍스트 표시
    }
});
