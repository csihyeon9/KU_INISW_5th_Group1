// sidepanel.js
document.addEventListener('DOMContentLoaded', () => {
    const contentDiv = document.getElementById('content');
    const loadingDiv = document.getElementById('loading');

    resetButton.addEventListener('click', () => {
        chrome.storage.local.remove(['lastAnalysis'], function() {
            contentDiv.innerHTML = '텍스트를 드래그하면 나타나는 아이콘을 클릭해주세요.';
        });
    });

    chrome.storage.local.get(['lastAnalysis'], function(result) {
        if (result.lastAnalysis) {
            contentDiv.innerHTML = result.lastAnalysis;
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
            contentDiv.classList.add('loading-active');  // 로딩 중 content 스타일 변경
            loadingDiv.style.display = 'block';

            const response = await fetch('http://localhost:8000/service', {
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
                const date = new Date(dateStr);
                return date.toLocaleString('ko-KR', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }).replace(/\. /g, '-').replace('.', '');
            };
            
            const analysisHTML = `
                <h3>분석 결과</h3>
                <p>${data.text}</p>
                <small>응답 시간: ${formatDate(data["response time"])}</small>
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
            contentDiv.classList.remove('loading-active');  // 로딩 완료 후 스타일 제거
        }
    }
});