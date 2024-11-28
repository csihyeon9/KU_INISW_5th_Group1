// sidepanel.js
document.addEventListener('DOMContentLoaded', () => {
    const contentDiv = document.getElementById('content');
    const loadingDiv = document.getElementById('loading');
    let initialMessage = '텍스트를 드래그하면 나타나는 아이콘을 클릭해주세요.';

    resetButton.addEventListener('click', () => {
        chrome.storage.local.remove(['lastAnalysis'], function() {
            contentDiv.innerHTML = initialMessage;
        });
    });

    chrome.storage.local.get(['lastAnalysis'], function(result) {
        if (result.lastAnalysis) {
            contentDiv.innerHTML = result.lastAnalysis;
        } else {
            contentDiv.innerHTML = initialMessage;
        }
    });

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "analyzeText") {
            sendResponse({ received: true });
            handleAnalysis(request.text);
            return false;
        }
    });

    async function handleAnalysis(text) {
        try {
            contentDiv.classList.add('loading-active');
            loadingDiv.style.display = 'block';
            contentDiv.innerHTML = ''; // 로딩 중에는 내용을 비움

            const response = await fetch('https://finwise.p-e.kr:8000/service', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received data:', data);
            
            const formatDate = (dateStr) => {
                const cleanedDateStr = dateStr
                    .replace('T', ' ')
                    .split('.')[0];
                return cleanedDateStr;
            };
            
            const analysisHTML = `
                <h3>분석 결과</h3>
                <p>${data.text}</p>
                <small>응답 시간: ${formatDate(data["response_time"])}</small>
            `;
            
            contentDiv.innerHTML = analysisHTML;

            chrome.storage.local.set({
                lastAnalysis: analysisHTML
            });

        } catch (error) {
            const errorHTML = `<div class="error">분석 중 오류가 발생했습니다: ${error.message}</div>`;
            contentDiv.innerHTML = errorHTML;
            chrome.storage.local.set({
                lastAnalysis: errorHTML
            });
            console.error('Analysis error:', error);
        } finally {
            loadingDiv.style.display = 'none';
            contentDiv.classList.remove('loading-active');
        }
    }
});